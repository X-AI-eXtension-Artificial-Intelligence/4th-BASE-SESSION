import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)  # 인코더 모듈 생성
        self.decoder = Decoder(params)  # 디코더 모듈 생성

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source)  # 인코더에 소스 문장을 입력하여 인코더의 출력을 계산
        # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)
        # 디코더에 타겟 문장, 소스 문장, 인코더 출력을 입력하여 디코더의 출력과 어텐션 맵 계산
        # output: [batch size, target length, output dim]
        # attn_map: [batch size, n heads, target length, source length]
        return output, attn_map

    def count_params(self):
        # 모델의 파라미터 수 계산
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
