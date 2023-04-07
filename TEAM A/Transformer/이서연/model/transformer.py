import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        ## encoder와 decoder 생성
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        ## encoder의 output을 얻음
        encoder_output = self.encoder(source)                            # [batch size, source length, hidden dim]
        ## decoder를 호출하여 output과 attention map 반환
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map
    
    ## 모델의 파라미터 수를 반환
    def count_params(self):
        ## 파라미터 리스트를 가져와서 각 파라미터 텐서의 원소 개수를 누적해서 더함
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
