import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight # model 폴더 -> ops -> init_weight 함수를 불러온다.


class MultiHeadAttention(nn.Module):
    def __init__(self, params, pre_trained=False):
        super(MultiHeadAttention, self).__init__() # nn.Module 클래스 속성을 모두 상속받음
        assert params.hidden_dim % params.n_head == 0 # n_head으로 hidden_dim을 나눴을 때 나누어떨어지지 않으면 오류를 발생
        self.pre_trained = pre_trained
        self.attentions = nn.ModuleList([SelfAttention(params, pre_trained=self.pre_trained)
                                         for _ in range(params.n_head)]) # SelfAttention을 n_head의 개수만큼 생성
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False) # 출력 형상을 조절하기 위한 선형 결합(hidden_dim, hidden_dim)
        
        if self.pre_trained:
            pass
        else:
            init_weight(self.o_w) # 가중치를 초기화한다
        
        self.dropout = nn.Dropout(params.dropout) # 드롭아웃을 적용(비율은 params에 존재)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions] # attention 계층에 파라미터 대입
        # self_attentions = [batch size, sentence length, attention dim] * num head
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]  # weighted_v 가 무슨 형태인지 궁금

        weighted_v = torch.cat(weighted_vs, dim=-1) # 계산된 attention을 하나로 합침
        # weighted_v = [batch size, sentence length, hidden dim]

        output = self.dropout(self.o_w(weighted_v)) # 형상 조절 후 드롭아웃 적용
        # output = [batch size, sentence length, hidden dim]

        return output, attentions # 출력과 어텐션 return


class SelfAttention(nn.Module): # SelfAttention을 구현
    def __init__(self, params, pre_trained=False):
        super(SelfAttention, self).__init__() # nn.Module 을 상속
        self.pre_trained = pre_trained
        self.hidden_dim = params.hidden_dim # hidden_dim 저장
        self.attention_dim = params.hidden_dim // params.n_head # attention_dim 저장

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) # query 가중치
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) # key 가중치
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False) # value 가중치
        
        if self.pre_trained:
            pass
        else:
            init_weight(self.q_w)
            init_weight(self.k_w)
            init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) # 스케일링 적용

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1)) # k와 q를 행렬 곱 하기 위해 k.T 를 수행
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf) # masking 여부 판단 뒤 masking 적용

        # normalize self attention score by applying soft max function on each row
        attention_score = F.softmax(self_attention, dim=-1) # softmax 함수를 사용해 self attention 점수를 정규화
        norm_attention_score = self.dropout(attention_score) 
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v) # 계산된 attention과 value와의 행렬 곱 계산
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
