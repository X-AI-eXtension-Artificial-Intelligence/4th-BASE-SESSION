import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module): #멀티헤드어텐션 -> 디코더에 해당
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0 #hidden_dim이 n_head로 나누어 떨어지는지 확인
        self.attentions = nn.ModuleList([SelfAttention(params) #n_head개의 self attention 모듈을 생성 => 멀티헤드어텐션 수행하는데 사용됨
                                         for _ in range(params.n_head)]) #8개의 attention
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False) #Linear 통해 출력 처리하는 선형계층 정의
        init_weight(self.o_w) 
        self.dropout = nn.Dropout(params.dropout) #드롭아웃진행

    def forward(self, query, key, value, mask=None): #input값으로 Q,K,V,mask가 들어옴
        # query, key, value = [batch size, sentence length, hidden dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions] #리스트에서 각각의 객체에 값 넣어서 나온 텐서들을 모아 놓은 것
        # self_attentions = [batch size, sentence length, attention dim] * num head
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions] # 배치 사이즈, 문장 길이, dim의 shape를 가진 텐서만 모아놓은 리스트
        attentions = [weighted_v[1] for weighted_v in self_attentions] # 배치 사이즈, 문장 길이 #attention score에 대한 값
        #weight들을 모아주는 리스트

        weighted_v = torch.cat(weighted_vs, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]

        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]

        return output, attentions


class SelfAttention(nn.Module): # self attention 
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head #multi head attention 8개
        # 512/8
        
        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None): #Q,K,V값을 input으로 받는다
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1)) #행렬 곱 수행
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf) #마스크를 적용할경우 마스크된 부분에 -inf

        # normalize self attention score by applying soft max function on each row
        attention_score = F.softmax(self_attention, dim=-1) #각 행에 대해 정규화된 어텐션 점수를 구함(시각화에 대한 -> 별로 신경x)
        norm_attention_score = self.dropout(attention_score) #드롭아웃
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v) #정규화된 attention 점수와 V행렬을 곱해서 가중치된 V 행렬을 계산
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
        
