import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module): ## 다수의 self-attention 모듈을 사용하여 입력된 Q, K, V를 각 head에 대해 연산하고 연산 결과를 합침
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0 ## 나누어 떨어지는지 검사 -> 나누어 떨어지지 않으면 에러 발생
        self.attentions = nn.ModuleList([SelfAttention(params)
                                         for _ in range(params.n_head)]) ## n_head만큼 SelfAttention 객체를 생성
        
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w) ## 가중치 초기화하는 함수 적용
        self.dropout = nn.Dropout(params.dropout) ## dropout layer

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        ## 각 head마다의 Self-Attention 결과를 저장
        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions = [batch size, sentence length, attention dim] * num head

        ## self_attentions 리스트에서 weighted_v와 attention_score 정보를 추출하여 리스트로 저장
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]

        ## 각 head에서 나온 weighted_v를 모아서 concat -> Multi-Head Attention의 최종 결과
        weighted_v = torch.cat(weighted_vs, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]

        ## weighted_v를 다시 hidden_dim 차원으로 변환한 후 dropout을 적용
        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim ## 입력 문장의 단어 벡터 차원
        self.attention_dim = params.hidden_dim // params.n_head ## self-attention에 사용할 Q, K, V 행렬의 차원 수

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) ## Q 행렬을 위한 선형 변환 layer
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) ## 입력 벡터의 크기를 attntion_dim으로 줄임임
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        ## 가중치 초기화
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout) ## 드롭아웃 layer
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) ## 어텐션 값이 큰 경우에 정규화하기 위한 상수

    def forward(self, query, key, value, mask=None): ## self-attention 계산
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        ## query, key, value를 이용해서 'q', 'k', 'v' 행렬을 구함
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        ## self-attention score 계산
        self_attention = torch.bmm(q, k.permute(0, 2, 1)) ## 내적
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        ## mask 적용
        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        # normalize self attention score by applying soft max function on each row
        ## self-attention score을 softmax에 적용
        attention_score = F.softmax(self_attention, dim=-1)
        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        ## 각 단어의 가중치를 구함
        weighted_v = torch.bmm(norm_attention_score, v) ## 가중치가 반영된 value 행렬
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
