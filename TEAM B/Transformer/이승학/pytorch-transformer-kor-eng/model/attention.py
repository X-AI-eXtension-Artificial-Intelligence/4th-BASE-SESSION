import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0
        self.attentions = nn.ModuleList([SelfAttention(params)                  # multi-head attention n_head만큼 modulelist에 저장 / (SelfAttention class)
                                         for _ in range(params.n_head)])
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)  # output weight 크기 지정과 weight 초기화
        init_weight(self.o_w)
        self.dropout = nn.Dropout(params.dropout)                               # dropout

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions = [batch size, sentence length, attention dim] * num head
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]          # self attention return값[0]
        attentions = [weighted_v[1] for weighted_v in self_attentions]           # self attention return값[1]

        weighted_v = torch.cat(weighted_vs, dim=-1)                              # multi-head attention 진행했던거 cat
        # weighted_v = [batch size, sentence length, hidden dim]

        output = self.dropout(self.o_w(weighted_v))                              # output weight를 곱해 최종 output값
        # output = [batch size, sentence length, hidden dim]

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim                                  # hidden_dim = 512 (논문)
        self.attention_dim = params.hidden_dim // params.n_head              # n_head = 8 / attention_dim = 64

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)# Q weight 
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)# K weight
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)# V weight        -> selfattention이라 Q, K, V에 같은 값이 들어감
        init_weight(self.q_w)                                                # Q, K, V init_weight
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)                            # dropout
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) #Scaled dot-product Attention을 위한 scale_factor

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1))                          # torch.bmm은 행렬 곱셈 / Q, K를 배치단위로 행렬 곱 실행
        self_attention = self_attention / self.scale_factor                        # Scaled dot-product Attention
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)             # masking

        # normalize self attention score by applying soft max function on each row
        attention_score = F.softmax(self_attention, dim=-1)                        # softmax로 self_attention 값의 attention score값을 얻음
        norm_attention_score = self.dropout(attention_score)                       # attention score에 대해 dropout진행 ??
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v)                            # norm_attention_score와 V와의 행렬곱
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
