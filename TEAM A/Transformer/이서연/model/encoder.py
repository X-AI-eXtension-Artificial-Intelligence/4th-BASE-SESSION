import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask):
        # source          = [batch size, source length, hidden dim] ## 입력 시퀀스의 단어 embedding vector
        # source_mask     = [batch size, source length, source length] ## 위치정보를 나타내는 마스크

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source) ## layer normalization
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0] ## self-attention 연산 후 원본 source와 더함 -> 잔차연결

        normalized_output = self.layer_norm(output) ## 다시 layer normalization
        output = output + self.position_wise_ffn(normalized_output) ## position-wise feedforward 연산결과와 이전 output을 더함 -> 잔차연결
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        ## Embedding layer / input_dim: 전체 단어 집합합 크기, hidden_dim: embedding 벡터 차원 수
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)
        ## Enbedding layer의 가중치 초기화
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        ## Embedding vector scaling 값 (hidden_dim의 제곱근)
        self.embedding_scale = params.hidden_dim ** 0.5
        ## positional Encoding
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        ## Encoder layer를 쌓음
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        ## dropout을 적용하기 위한 객체
        self.dropout = nn.Dropout(params.dropout)
        ## layer normalization
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, source):
        # source = [batch size, source length]
        ## source mask 생성
        source_mask = create_source_mask(source)      # [batch size, source length, source length]
        ## positional encoding이 적용된 vector
        source_pos = create_position_vector(source)   # [batch size, source length]

        ## 입력 시퀀스의 단어들을 embedding vector로 변환 후 scaling
        source = self.token_embedding(source) * self.embedding_scale
        ## positional encoding 적용 후 dropout
        source = self.dropout(source + self.pos_embedding(source_pos))
        # source = [batch size, source length, hidden dim]

        ## Encoderlayer 반복
        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)
        # source = [batch size, source length, hidden dim]

        ## layer normalization 적용 후 반환
        return self.layer_norm(source)
