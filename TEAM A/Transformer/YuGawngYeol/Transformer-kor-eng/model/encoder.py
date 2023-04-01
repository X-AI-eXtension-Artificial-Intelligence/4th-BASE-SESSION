import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) # Layer norm 설정
        self.self_attention = MultiHeadAttention(params) # 함수 이동 -> multi headattention 지정
        self.position_wise_ffn = PositionWiseFeedForward(params) #함수 이동 -> PositionWiseFeedForward 설정 

    def forward(self, source, source_mask):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source) #norm 적용
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0] # attention + residual 

        normalized_output = self.layer_norm(output) # norm
        output = output + self.position_wise_ffn(normalized_output) #ffn 적용
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx) #embedding 지정
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim ** 0.5 # embedding_scale 생성 -> 규제항
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True) # 함수 이동 -> positional encoding 생성

        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)]) # 함수 이동 -> N_later만큼 encoder Layer 생성
        self.dropout = nn.Dropout(params.dropout) # 드롭 아웃
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) # LayerNorm

    def forward(self, source):
        # source = [batch size, source length]
        source_mask = create_source_mask(source)      # [batch size, source length, source length] -> 함수 이동 -> pad 마스크 처리
        source_pos = create_position_vector(source)   # [batch size, source length] # 함수 이동 -> position 벡터 생성

        source = self.token_embedding(source) * self.embedding_scale
        source = self.dropout(source + self.pos_embedding(source_pos)) # source 생성 embedding  + position 
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask) # later 만큼 진행
        # source = [batch size, source length, hidden dim]

        return self.layer_norm(source)
