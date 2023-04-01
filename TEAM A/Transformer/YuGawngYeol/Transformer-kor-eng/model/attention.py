import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0 # 조건에 맞지 않으면 에러 발생
        self.attentions = nn.ModuleList([SelfAttention(params) # n_head 갯수 만큼 self attention 생성 -> 함수 이동
                                         for _ in range(params.n_head)]) # attention 생성
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False) # Linear layer 생성 
        init_weight(self.o_w) # -> 가중치 초기화
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions] # modulelist에 attetin 수 만큼 -> n_head
        # self_attentions = [batch size, sentence length, attention dim] * num head
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions] # weight_vs
        attentions = [weighted_v[1] for weighted_v in self_attentions] # attention_score

        weighted_v = torch.cat(weighted_vs, dim=-1)  # weight_vs끼리 결합
        # weighted_v = [batch size, sentence length, hidden dim]

        output = self.dropout(self.o_w(weighted_v)) # Linear 거쳐서 output
        # output = [batch size, sentence length, hidden dim]

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim 
        self.attention_dim = params.hidden_dim // params.n_head # attention 차원 생성

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        init_weight(self.q_w) # -> 함수 이동
        init_weight(self.k_w) # 가중치 초기화
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout) # dropout 설정
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) #scale_factor 생성 -> root d 설정

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        # 동일한 입력 문장을 사용하여 Q, K, V 행렬을 만들어 자기 주의 점수를 계산합니다
        q = self.q_w(query) # q,k,v 생성 
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1)) #q*k k -> permute를 통해 transpose
        self_attention = self_attention / self.scale_factor # root d로 나눔
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None: # mask가 있으면
            self_attention = self_attention.masked_fill(mask, -np.inf) #mask -inf값 생성

        # normalize self attention score by applying soft max function on each row
        # 각 행에 소프트 최대 함수를 적용하여 자체 주의 점수를 정규화합니다
        attention_score = F.softmax(self_attention, dim=-1) # soft max 진행
        norm_attention_score = self.dropout(attention_score) # drop out
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        # 자체 주의 점수와 V 행렬을 사용하여 "가중치" 값 행렬을 계산합니다
        weighted_v = torch.bmm(norm_attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
