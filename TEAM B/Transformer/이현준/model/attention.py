import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__() # nn.Module의 init 상속
        assert params.hidden_dim % params.n_head == 0 # hidden_dim을 n_head로 나눈 나머지가 0이 아니면 AssertError 발생
        self.attentions = nn.ModuleList([SelfAttention(params) for _ in range(params.n_head)]) 
        # n_head만큼 SelfAttention을 만들어 nn.ModuleList에 저장해서 attentions에 저장
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        # hidden_dim 차원의 input을 hidden_dim 차원으로 output하는 선형회귀를 한 뒤 o_w에 저장 : Multi-Head Attention 시행 후 스킵커넥션하기 전 형상을 맞춰줌
        init_weight(self.o_w) # o_w의 가중치 초기화
        self.dropout = nn.Dropout(params.dropout) # 드롭아웃 실행

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions = [batch size, sentence length, attention dim] * num head
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]

        weighted_v = torch.cat(weighted_vs, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]

        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__() # nn.Module의 생성자 오버라이딩
        self.hidden_dim = params.hidden_dim # word vector dimension
        self.attention_dim = params.hidden_dim // params.n_head # 

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) # hidden_dim의 input으로 attention_dim의 output을 생성하는 linearRegression -> q_w
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) # hidden_dim의 input으로 attention_dim의 output을 생성하는 linearRegression -> k_w
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) # hidden_dim의 input으로 attention_dim의 output을 생성하는 linearRegression -> v_w
        init_weight(self.q_w) # 가중치 초기화
        init_weight(self.k_w) # 가중치 초기화
        init_weight(self.v_w) # 가중치 초기화

        self.dropout = nn.Dropout(params.dropout) # 드롭아웃
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) # attention_dim의 제곱근 -> scale_factor

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1)) # q와 permute한 k를 배치행렬곱
        self_attention = self_attention / self.scale_factor # 스케일링
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None: # 마스크가 있으면
            self_attention = self_attention.masked_fill(mask, -np.inf) # self_attention에서 mask된 값을 음의 무한대로 마스킹

        # normalize self attention score by applying soft max function on each row
        attention_score = F.softmax(self_attention, dim=-1) # self_attention에 softmax 적용
        norm_attention_score = self.dropout(attention_score) # 드롭아웃 적용
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
