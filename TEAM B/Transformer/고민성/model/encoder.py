import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) # hidden_dim 512
        self.self_attention = MultiHeadAttention(params) # 셀프 어텐션 정의
        self.position_wise_ffn = PositionWiseFeedForward(params)# 포지션와이즈 피드포워드 정의

    def forward(self, source, source_mask):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        # 논문이랑 다른 점? 정규화 진행 후 self_attention에 들어감
        normalized_source = self.layer_norm(source) # 정규화 진행
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]
        # x + SubLayer(x) 표현 [0]번째 인덱스.

        normalized_output = self.layer_norm(output)
        # 정규화 진행

        output = output + self.position_wise_ffn(normalized_output)
        #output =  x + SubLayer(LayerNorm(x))+ SubLayer2(LayerNorm(x + SubLayer(LayerNorm(x))))
        # 이럴경우 normalization 이 한번더 진행되어짐.
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)
        # 인풋 임베딩
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim ** 0.5
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)
        # 포지셔널 인코딩

        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        # 인코더 레이어 정의, config 파일이 n_layer 6개로 정의

        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, source):
        # source = [batch size, source length]
        source_mask = create_source_mask(source)      # [batch size, source length, source length]
        # 인풋에 대한 마스킹

        source_pos = create_position_vector(source)   # [batch size, source length]
        # 인풋에 대한 포지셔널 인코딩

        source = self.token_embedding(source) * self.embedding_scale
        # 인풋 데이터 임베딩
        
        source = self.dropout(source + self.pos_embedding(source_pos))
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)
        # source = [batch size, source length, hidden dim]

        return self.layer_norm(source)
