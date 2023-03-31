import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)#encoder 정의
        self.decoder = Decoder(params)#decoder 정의

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source)                            # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map
        # output 말 그대로 결과값, attn_map은 디코더의 인풋 쿼리와 인코더의 키간의 관계 맵
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
