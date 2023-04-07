import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()

        # n_head로 나누어 떨어지는 hidden_dim인지 확인합니다.
        assert params.hidden_dim % params.n_head == 0

        # n_head 개수 만큼 SelfAttention 모듈을 만들어 nn.ModuleList로 저장합니다.
        self.attentions = nn.ModuleList([SelfAttention(params) for _ in range(params.n_head)])
        
        # 출력을 위한 linear layer와 dropout을 정의합니다.
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w)
        self.dropout = nn.Dropout(params.dropout)
# MultiHeadAttention 클래스를 정의합니다.
# params는 파라미터들을 담은 dictionary입니다.
# super()로 nn.Module을 초기화합니다.
# params.hidden_dim은 MultiHeadAttention의 입력 크기입니다.
# params.n_head는 multi-head attention에서 head의 수를 의미합니다.
# assert를 사용하여 입력 크기가 head의 수로 나누어 떨어지는지 확인합니다.
# self.attentions는 multi-head attention 내의 각 head를 담은 리스트입니다.
# nn.ModuleList()로 리스트를 모듈 형태로 감쌉니다.
# SelfAttention 클래스를 params.n_head만큼 호출하여 self.attentions에 추가합니다.
# self.o_w는 multi-head attention의 출력을 만들기 위한 fully connected layer입니다.
# init_weight() 함수를 사용하여 self.o_w의 가중치를 초기화합니다.
# self.dropout은 입력값의 일부를 무작위로 0으로 바꾸어 regularization을 수행하는 dropout 레이어입니다.


    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # 각 SelfAttention 모듈에 query, key, value를 전달하고 결과값들을 리스트에 저장합니다.
        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions = [batch size, sentence length, attention dim] * num head

        # 각 SelfAttention 모듈에서 계산한 결과값 중, 가중평균된 값들을 추출합니다.
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]

        # 각 SelfAttention 모듈에서 계산한 결과값 중, 어텐션 스코어를 추출합니다.
        attentions = [weighted_v[1] for weighted_v in self_attentions]

        # 가중평균된 값을 concatenate하여 multi-head self-attention 결과값을 만듭니다.
        weighted_v = torch.cat(weighted_vs, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]

        # 출력값에 dropout과 linear layer를 적용합니다.
        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]

        # 출력값과 어텐션 스코어를 반환합니다.
        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()

        # hidden_dim을 n_head로 나눈 값을 attention_dim으로 설정합니다.
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head

        # Q, K, V matrix를 위한 linear layer를 정의합니다.
        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)

        # weight 초기화를 위한 함수를 호출합니다.
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        # dropout을 정의합니다.
        self.dropout = nn.Dropout(params.dropout)

        # scale_factor를 계산하기 위해 attention_dim의 제곱근 값을 저장합니다.
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None):
        # 셀프 어텐션 연산 수행
        # query, key, value = [배치 크기, 문장 길이, 은닉 차원]

        # 입력 문장으로 Q, K, V 행렬 생성
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [배치 크기, 문장 길이, 어텐션 차원]

        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        # normalize self attention score by applying soft max function on each row
        attention_score = F.softmax(self_attention, dim=-1)
        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        weighted_v = torch.bmm(norm_attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
