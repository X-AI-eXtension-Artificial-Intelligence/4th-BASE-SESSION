import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector


class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.encoder_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # target          = [batch size, target length, hidden dim]
        # encoder_output  = [batch size, source length, hidden dim]
        # target_mask     = [batch size, target length, target length]
        # dec_enc_mask    = [batch size, target length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        norm_target = self.layer_norm(target)
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]

        # In Decoder stack, query is the output from below layer and key & value are the output from the Encoder
        norm_output = self.layer_norm(output)
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask)
        output = output + sub_layer
        # query는 decoder에서 key, value는 encoder의 output을 가져와서 사용
        # 여기서는 인덱싱 0을 하지 않아서, 결과값과 query, key의 representation값도 가져옴 (attn_map)
        norm_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(norm_output)
        # output = [batch size, target length, hidden dim]

        return output, attn_map


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim ** 0.5
        # 토큰 임베딩 고정
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)
        # 포지셔널 인코딩

        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, target, source, encoder_output):
        # target              = [batch size, target length]
        # source              = [batch size, source length]
        # encoder_output      = [batch size, source length, hidden dim]
        target_mask, dec_enc_mask = create_target_mask(source, target)
        # 패딩 부분에 대한 마스크 진행, 예측하는 구간 이후 마스킹 진행
        # target_mask / dec_enc_mask  = [batch size, target length, target/source length]
        target_pos = create_position_vector(target)  # [batch size, target length]
        # token에 대해서 POSITION INDEXING

        target = self.token_embedding(target) * self.embedding_scale
        # target embeding + positional embeding
        # token 임베딩 진행
        target = self.dropout(target + self.pos_embedding(target_pos))
        # target = [batch size, target length, hidden dim]

        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)
        # target = [batch size, target length, hidden dim]

        target = self.layer_norm(target)
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))
        # 최종 tarket을 위한 임베딩 진행
        # output = [batch size, target length, output dim]
        return output, attention_map
