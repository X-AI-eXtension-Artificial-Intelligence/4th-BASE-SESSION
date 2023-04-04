import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()                        
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)           #layerNorm
        self.self_attention = MultiHeadAttention(params)                      #encoder 앞의 self_attention
        self.position_wise_ffn = PositionWiseFeedForward(params)              #FeedForwardNetwork

    def forward(self, source, source_mask):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source)                           # layerNorm(source)
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0] # self_attention + source  /  Add & Norm

        normalized_output = self.layer_norm(output)                           # LayerNorm(output)
        output = output + self.position_wise_ffn(normalized_output)           # output + position_wise_ffn(output)  /  Add & Norm
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx) # input을 512차원의 hidden_dimention에 embedding
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)                    # normalization
        self.embedding_scale = params.hidden_dim ** 0.5                                                      # embedding_scale은 hidden_dim ** .5
        self.pos_embedding = nn.Embedding.from_pretrained(                                                   # positional embedding - ops.py의 create_positional_encoding
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])           # encoderlayer class의 modulelist 생성 n_layer = 6
        self.dropout = nn.Dropout(params.dropout)                                                            # dropout 지정
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)                                          # LayerNorm 지정

    def forward(self, source):
        # source = [batch size, source length]
        source_mask = create_source_mask(source)      # [batch size, source length, source length]           # ops.py의 create_source_mask
        source_pos = create_position_vector(source)   # [batch size, source length]                          # ops.py의 create_position_vector
 
        source = self.token_embedding(source) * self.embedding_scale                                         # embedding된 input과 embedding_scale을 곱함
        source = self.dropout(source + self.pos_embedding(source_pos))                                       # positional embedding + source 후 dropout
        # source = [batch size, source length, hidden dim]

        for encoder_layer in self.encoder_layers:                                                            # 6개의 encoder_layer에 대해 진행
            source = encoder_layer(source, source_mask)                                                      # source_mask값은 encoder에서 신경 안써도 됨
        # source = [batch size, source length, hidden dim]

        return self.layer_norm(source)
