import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight

# 일반적으로 Pytorch 모델을 사용할 때 torch.nn.Module을 상속
class MultiHeadAttention(nn.Module):

    # 모델에서 사용하는 module, activation function 등을 정의
    def __init__(self, params):

        # 자식 클래스인 MultiHeadAttention에 nn.Module을 불러오겠다는 뜻
        # python3는 super().__init__() 가능
        super(MultiHeadAttention, self).__init__()

        # AssertionError : 코드 작성자가 보증하지 않은 동작
        # embedding vector를 head의 개수만큼 나누어 주기 때문에 나누어 떨어지지 않으면 에러 발생
        assert params.hidden_dim % params.n_head == 0

        # nn.ModulList : 모듈을 리스트 형태로 저장 -> 추후에 모듈에 인덱스로 접근 가능
        # 하지만 nn.Sequential과 다르게 foward method가 없고 module간 connection도 없다.
        # 여기서는 head의 개수만큼 parameter를 SelfAttention한 결과를 리스트로 저장
        self.attentions = nn.ModuleList([SelfAttention(params)
                                         for _ in range(params.n_head)])

        # hidden layer의 크기만큼 fc layer 계산                                         
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        
        # 가중치 초기화
        init_weight(self.o_w)

        # Dropout 진행
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # attention을 진행한 결과를 head마다 리스트로 저장 
        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions = [batch size, sentence length, attention dim] * num head
        # Attention함수 수행 결과 : self.dropout(weighted_v), attention_score
        
        # Attention을 거친 weight들
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]
        # attention score들
        attentions = [weighted_v[1] for weighted_v in self_attentions]

        # 합침
        weighted_v = torch.cat(weighted_vs, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]

        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]

        return output, attentions

# Scaled Dot-Product Attention 수행 
class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)

        # 가중치 초기화
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)

        # Attention Score를 구하기 위해 유사도에 나눠줄 값 (root(d_k))
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        # Batch matrix multiplication, 두 행렬의 행렬 곱 (두 입력 모두 3-D tensor일 때 사용)
        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf) # pytorch함수 masked_fill, mask -> 음의 무한대로 변경경

        # normalize self attention score by applying soft max function on each row
        # Attention Score = softmax(QK^T/root(d_k)*V)
        attention_score = F.softmax(self_attention, dim=-1)
        
        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
