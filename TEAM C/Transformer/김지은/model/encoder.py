import torch.nn as nn

from model.attention import MultiHeadAttention #멀티헤드 어텐션 사용
from model.positionwise import PositionWiseFeedForward #positional feedward 신경망 사용
from model.ops import create_positional_encoding, create_source_mask, create_position_vector
#positional encoding, mask 생성, position vector 생성


class EncoderLayer(nn.Module): #인코더 레이어층
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source) 
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]
        # self attention 결과와 원래 source를 더해줌 => residual connection

        normalized_output = self.layer_norm(output) #더해진 결과를 layer normalization 진행
        output = output + self.position_wise_ffn(normalized_output) #결과값을 feedforward에 통과시킨 후, 이전단계에서의 결과를 더해줌
        # output = [batch size, source length, hidden dim]

        return output #반복 


class Encoder(nn.Module): #인코더 모델 구현
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx) #입력 토큰들을 임베딩 벡터로 변환
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5) #초기화 진행
        self.embedding_scale = params.hidden_dim ** 0.5
        #루트 d로 나눠주는 -> 크기 맞추려고 루트 0.5로해줌
        self.pos_embedding = nn.Embedding.from_pretrained( #positional encoding embedding 구현 (위치 정보 전달용도)
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)]) #param_layer 수만큼 쌓아 올리는 과정 수행
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) #마지막 출력값에 norm 진행

    def forward(self, source):
        # source = [batch size, source length]
        source_mask = create_source_mask(source)      # [batch size, source length, source length]
        #source(=input) 패딩 인덱스를 채워주는 역할
        
        source_pos = create_position_vector(source)   # [batch size, source length] #문장 위치한 위치 값 나타내는 값

        source = self.token_embedding(source) * self.embedding_scale #임베딩
        #?
        
        source = self.dropout(source + self.pos_embedding(source_pos)) #임베딩값(sourc) + 위치 인코딩 벡터
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers: # 반복
            source = encoder_layer(source, source_mask) # 각 6개 레이어 반복해서 통과
        # source = [batch size, source length, hidden dim]

        return self.layer_norm(source) # 반환
