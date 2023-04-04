import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector


class DecoderLayer(nn.Module): #decoderlayer 생성
    def __init__(self, params): #값 초기화
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.encoder_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, target, encoder_output, target_mask, dec_enc_mask): # input으로 들어갈 값들
        # target          = [batch size, target length, hidden dim]
        # encoder_output  = [batch size, source length, hidden dim]
        # target_mask     = [batch size, target length, target length]
        # dec_enc_mask    = [batch size, target length, source length]

        # Original Implementation: LayerNorm(x + SubLayer(x)) -> Updated Implementation: x + SubLayer(LayerNorm(x))
        norm_target = self.layer_norm(target) #레이어 정규화를 적용한 타겟값
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]
        # 셀프 어텐션 적용

        # In Decoder stack, query is the output from below layer and key & value are the output from the Encoder
        norm_output = self.layer_norm(output)
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask) # 결과값 반환
        #이전 층에서 출력된 값: Q, 인코더 출력값: Key, value -> 어텐션 진행
        output = output + sub_layer

        norm_output = self.layer_norm(output) # norm 적용
        output = output + self.position_wise_ffn(norm_output) #순전파 진행
        # output = [batch size, target length, hidden dim]

        return output, attn_map


class Decoder(nn.Module): #디코더 클래스
    def __init__(self, params): #초기화
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx) #임베딩
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5) #가중치 초기화
        self.embedding_scale = params.hidden_dim ** 0.5 #임베딩 벡터 스케일링값
        self.pos_embedding = nn.Embedding.from_pretrained( #위치 임베딩 
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True) # 만들어진 임베딩 벡터 사용 #freeze=True: 가중치 변하지 않도록

        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)]) #디코더 블럭 층 쌓은 리스트
        self.dropout = nn.Dropout(params.dropout) #드롭아웃
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) #정규화

    def forward(self, target, source, encoder_output): # decoder forward함수 (번역할 문장, 원본 문장, 인코더 출력 받음)
        # target              = [batch size, target length]
        # source              = [batch size, source length]
        # encoder_output      = [batch size, source length, hidden dim]
        target_mask, dec_enc_mask = create_target_mask(source, target)#mask생성 ( 번역하고자 하는 문장 (target), 원본 문장(source) 간의 mask)
        # target_mask / dec_enc_mask  = [batch size, target length, target/source length]
        target_pos = create_position_vector(target)  # [batch size, target length] #위치벡터 생성

        target = self.token_embedding(target) * self.embedding_scale #임베딩
        target = self.dropout(target + self.pos_embedding(target_pos)) # 임베딩 값+위치정보부여
        # target = [batch size, target length, hidden dim]

        for decoder_layer in self.decoder_layers: #decoder 6번 반복
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)
        # target = [batch size, target length, hidden dim]

        target = self.layer_norm(target) #결과값을 norm 
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))
        # output= decoder 끝나고 나온 결과값
        # 
        # output = [batch size, target length, output dim]
        return output, attention_map #반환
