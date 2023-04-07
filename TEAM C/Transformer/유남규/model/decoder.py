import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector

class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        # 입력으로 받은 params를 이용하여 모델 구성을 위한 초기화 수행
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)  # 레이어 정규화
        self.self_attention = MultiHeadAttention(params)  # 디코더 내부에서의 멀티 헤드 어텐션
        self.encoder_attention = MultiHeadAttention(params)  # 디코더에서 인코더와의 멀티 헤드 어텐션
        self.position_wise_ffn = PositionWiseFeedForward(params)  # 포지션 와이즈 피드포워드 네트워크

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # target : [batch size, target length, hidden dim] 디코더에서 생성된 이전 토큰 시퀀스
        # encoder_output : [batch size, source length, hidden dim] 인코더에서 생성된 인코딩된 소스 문장
        # target_mask : [batch size, target length, target length] 타겟 마스킹, 패딩 토큰 위치를 나타냄
        # dec_enc_mask : [batch size, target length, source length] 디코더-인코더 마스킹, 인코더 문장의 패딩 토큰 위치를 나타냄

        # 디코더 내부의 멀티 헤드 어텐션을 수행하여 새로운 토큰을 생성
        # LayerNorm(x + SubLayer(x)) 형태로 구성
        # norm_target: [batch size, target length, hidden dim]
        norm_target = self.layer_norm(target)
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]

        # 디코더와 인코더 간의 멀티 헤드 어텐션을 수행하여 새로운 토큰을 생성
        # 인코더에서 생성된 어텐션 맵을 사용
        norm_output = self.layer_norm(output)
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask)
        output = output + sub_layer

        # 포지션 와이즈 피드포워드 네트워크를 수행하여 새로운 토큰을 생성
        norm_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(norm_output)
        # output: [batch size, target length, hidden dim]

        return output, attn_map

class Decoder(nn.Module):
#     위 코드에서 Decoder 클래스는 다음과 같은 기능을 가지고 있습니다.

# 입력된 target 시퀀스를 임베딩 레이어에 입력하여 임베딩된 텐서를 얻습니다.
# 포지셔널 인코딩을 수행하여 임베딩된 텐서에 더해줍니다.
# 임베딩된 텐서에 드롭아웃을 수행합니다.
# 디코더 레이어들을 차례대로 거치면서 디코딩을 수행합니다.
# 디코더의 출력값을 최종 출력 레이어
    def __init__(self, params):
        # 입력으로 받은 params를 이용하여 모델 구성을 위한 초기화 수행
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim ** 0.5
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, target, encoder_output, src_mask, tgt_mask):
        # target : [batch size, target length] 디코더의 입력 시퀀스
        # encoder_output : [batch size, source length, hidden dim] 인코더에서 생성된 인코딩된 소스 문장
        # src_mask : [batch size, source length] 인코더의 패딩 토큰 위치를 마스킹한 텐서
        # tgt_mask : [batch size, target length] 디코더의 패딩 토큰 위치와 미래 시점의 정보를 마스킹한 텐서

        target_mask, dec_enc_mask = create_target_mask(source, target)
        # target_mask / dec_enc_mask  = [batch size, target length, target/source length]
        target_pos = create_position_vector(target)  # [batch size, target length]

        target = self.token_embedding(target) * self.embedding_scale
        target = self.dropout(target + self.pos_embedding(target_pos))
        # target = [batch size, target length, hidden dim]

        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)
        # target = [batch size, target length, hidden dim]

        target = self.layer_norm(target)
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))
        # output = [batch size, target length, output dim]
        return output, attention_map
