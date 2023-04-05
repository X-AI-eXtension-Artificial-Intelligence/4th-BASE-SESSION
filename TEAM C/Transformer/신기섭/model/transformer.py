import torch.nn as nn

from model.encoder import Encoder # model 폴더 -> encoder -> Encoder를 불러온다.
from model.decoder import Decoder # model 폴더 -> decoder -> Decoder를 불러온다.


class Transformer(nn.Module):
    def __init__(self, params, pre_trained=False):
        super(Transformer, self).__init__()
        self.pre_trained = pre_trained
        self.encoder = Encoder(params, pre_trained=self.pre_trained)
        self.decoder = Decoder(params, pre_trained=self.pre_trained)

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source)                            # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map

    def count_params(self): # 사용되는 총 파라미터의 개수를 출력
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
