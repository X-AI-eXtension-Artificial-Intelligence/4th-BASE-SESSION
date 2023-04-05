import torch.nn as nn

from model.attention import MultiHeadAttention # model 폴더 -> attention -> MultiHeadAttention 클래스를 불러온다.
from model.positionwise import PositionWiseFeedForward # model 폴더 -> positionwise -> PositionWiseFeedForward 를 불러온다.
from model.ops import create_positional_encoding, create_source_mask, create_position_vector # model 폴더 -> ops -> 각종 함수 불러온다.


class EncoderLayer(nn.Module):
    def __init__(self, params, pre_trained=False):
        self.pre_trained = pre_trained
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) # 정규화 계층
        self.self_attention = MultiHeadAttention(params, pre_trained=self.pre_trained) 
        self.position_wise_ffn = PositionWiseFeedForward(params, pre_trained=self.pre_trained)

    def forward(self, source, source_mask):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source)
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]

        normalized_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(normalized_output)
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params, pre_trained=False):
        super(Encoder, self).__init__()
        self.pre_trained=pre_trained
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx) # 각 단어에 해당하는 임베딩 벡터 출력
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5) # 임베딩 계층에서의 가중치 초기화
        self.embedding_scale = params.hidden_dim ** 0.5 # 스케일링을 위한 변수
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True) # positional encoding 생성

        self.encoder_layers = nn.ModuleList([EncoderLayer(params, pre_trained=self.pre_trained) for _ in range(params.n_layer)]) # 인코더 레이어 n_layer 개 생성
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, source):
        # source = [batch size, source length]
        source_mask = create_source_mask(source)      # [batch size, source length, source length]
        source_pos = create_position_vector(source)   # [batch size, source length]

        source = self.token_embedding(source) * self.embedding_scale
        source = self.dropout(source + self.pos_embedding(source_pos))
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)
        # source = [batch size, source length, hidden dim]

        return self.layer_norm(source) # 인코더를 지정한 횟수만큼 반복 후, 정규화 계층을 return
