import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        # source = [batch size, source length] : encoder 입력
        # target = [batch size, target length] : decoder 입력
        encoder_output = self.encoder(source)                            # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map

    #모델 파라미터 개수 계산
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
