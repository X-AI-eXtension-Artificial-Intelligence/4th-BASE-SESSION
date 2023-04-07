import torch.nn as nn
from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector

class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        # 입력 파라미터 params에 따라 모델을 초기화합니다.
        # LayerNorm 모듈을 정의합니다.
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        # Multi-Head Attention 모듈을 정의합니다.
        self.self_attention = MultiHeadAttention(params)
        # Position-Wise Feed Forward Network 모듈을 정의합니다.
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask):
        # source = [batch size, source length, hidden dim]
        # source_mask = [batch size, source length, source length]

        # 입력으로 들어온 source를 LayerNorm 모듈을 통해 정규화합니다.
        normalized_source = self.layer_norm(source)
        # Multi-Head Attention 모듈을 통해 self-attention 값을 계산합니다.
        # query, key, value는 모두 정규화된 source입니다.
        # source_mask는 source에 패딩을 표현한 것으로, 패딩 토큰에 attention이 가지 않도록 하기 위해 사용됩니다.
        # output은 self-attention 값과 source의 더한 값입니다.
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]

        # 이전 출력값에 LayerNorm 모듈을 통해 정규화합니다.
        normalized_output = self.layer_norm(output)
        # Position-Wise Feed Forward Network 모듈을 통해 값을 계산합니다.
        output = output + self.position_wise_ffn(normalized_output)
        # output = [batch size, source length, hidden dim]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        # 입력 파라미터 params에 따라 모델을 초기화합니다.
        # token_embedding은 입력으로 들어온 단어의 임베딩 벡터를 구합니다.
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)
        # token_embedding의 weight를 정규 분포로 초기화합니다.
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        # positional encoding 값을 구하는 데 사용될 임베딩 스케일 변수를 정의합니다.
        self.embedding_scale = params.hidden_dim ** 0.5
        # 입력 문장의 위치 정보를 임베딩하는 모듈을 정의합니다.
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)
        # EncoderLayer를 params.n_layer개 만큼 반복하여 쌓습니다.
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        # 입력을 드롭아웃합니다.
        self.dropout = nn.Dropout(params.dropout)
        # 입력 값을 정규
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, source):
        # source = [batch size, source length]
        # 소스 마스크 생성
        source_mask = create_source_mask(source)      # [batch size, source length, source length]
        # 소스 위치 정보 생성
        source_pos = create_position_vector(source)   # [batch size, source length]

        # 입력 토큰 임베딩
        source = self.token_embedding(source) * self.embedding_scale
        # 입력 토큰 임베딩에 위치 정보 임베딩을 더해서 입력 소스 텐서 생성
        source = self.dropout(source + self.pos_embedding(source_pos))
        # source = [batch size, source length, hidden dim]

        # 인코더 층에 입력 소스 텐서를 입력해서 인코딩
        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)
        # source = [batch size, source length, hidden dim]

        # 인코딩된 소스를 정규화
        return self.layer_norm(source)