import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module): # 그동안 했던 거 한꺼번에 실행
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source) # encoder-decoder attention에 들어가는 그거     # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map # encoder-decoder attention의 출력

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) # 모델 전체 파라미터 개수
