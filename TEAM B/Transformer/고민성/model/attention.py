import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0
        self.attentions = nn.ModuleList([SelfAttention(params)
                                         for _ in range(params.n_head)])
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w)
        self.dropout = nn.Dropout(params.dropout)

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
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head  # head의 수에 따라 나눔  hidden_dim / head = 64

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        init_weight(self.q_w)# 가중치 이니셜라이즈 사비에르 분포
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) # 가중치 정규화를 위해 구하는거

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        q = self.q_w(query) #linear layer진행
        k = self.k_w(key) #linear layer진행
        v = self.v_w(value) #linear layer진행
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1)) # 행렬 곱 진행
        self_attention = self_attention / self.scale_factor # 1/루트 ( hidden_dim )
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)# 마스킹하기

        # normalize self attention score by applying soft max function on each row
        attention_score = F.softmax(self_attention, dim=-1) #소프트 맥스 함수 적용
        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v) # self_attention 행렬 곱(소프트맥스(행렬 곱(query*key))*value)
        # weighted_v = [batch size, sentence length, attention dim]
        
        
        # 두가지를 내보내는 이유
        # query*key 자체가 loss 산정을 위함
        # query*key*value는 결과값 도출
        return self.dropout(weighted_v), attention_score
