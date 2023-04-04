import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector


class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)                        # LayerNorm
        self.self_attention = MultiHeadAttention(params)                                   # MultiHeadAttention
        self.encoder_attention = MultiHeadAttention(params)                                # MultiHeadAttention
        self.position_wise_ffn = PositionWiseFeedForward(params)                           # PositionWiseFeedForward

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # target          = [batch size, target length, hidden dim]
        # encoder_output  = [batch size, source length, hidden dim]
        # target_mask     = [batch size, target length, target length]
        # dec_enc_mask    = [batch size, target length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        norm_target = self.layer_norm(target)                                                                       # target값을 LayerNorm
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]                # target값에 대해 self_attention을 진행하고, 기존 target값에 더해줌

        # In Decoder stack, query is the output from below layer and key & value are the output from the Encoder
        norm_output = self.layer_norm(output)                                                                       # 위에서 구한 output layerNorm
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask)  
                                                                                                                    #sub_layer와 attention_map을 encoder_attention(multi-head attention)을 사
                                                                                                                    #용해 얻음 이 때 사용되는 k,v값은 encoder에서 q값은 decoder의 이전 layer에서 
                                                                                                                    #얻게됨
                                                                                                                    
        output = output + sub_layer                                                                                 # Add&Norm

        norm_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(norm_output)                                                       # PositionWiseFeedForward
        # output = [batch size, target length, hidden dim]

        return output, attn_map


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)  # decoder_input을 512차원의 hidden_dimention에 embedding
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)                      # normalization
        self.embedding_scale = params.hidden_dim ** 0.5                                                        # embedding_scale은 hidden_dim ** .5
        self.pos_embedding = nn.Embedding.from_pretrained(                                                     # positional embedding - ops.py의 create_positional_encoding
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)                      

        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])             # decoderlayer class의 modulelist 생성 n_layer = 6
        self.dropout = nn.Dropout(params.dropout)                                                              # dropout 지정
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)                                            # layerNorm 지정

    def forward(self, target, source, encoder_output):
        # target              = [batch size, target length]
        # source              = [batch size, source length]
        # encoder_output      = [batch size, source length, hidden dim]
        target_mask, dec_enc_mask = create_target_mask(source, target)                                         # ops.py의 create_target_mask
        # target_mask / dec_enc_mask  = [batch size, target length, target/source length]
        target_pos = create_position_vector(target)  # [batch size, target length]                             # ops.py의 create_position_vector

        target = self.token_embedding(target) * self.embedding_scale                                           # embedding된 target과 embedding_scale을 곱함
        target = self.dropout(target + self.pos_embedding(target_pos))                                         # target + positional embedding 후 dropout
        # target = [batch size, target length, hidden dim]

        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)           # 6개의 decoder_layer에 대해 진행
        # target = [batch size, target length, hidden dim]

        target = self.layer_norm(target)                                                                       # layerNorm(target)
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))                             # 처음의 token_embedding weigth와 행렬곱을 통해 output 반환
        # output = [batch size, target length, output dim]
        return output, attention_map
