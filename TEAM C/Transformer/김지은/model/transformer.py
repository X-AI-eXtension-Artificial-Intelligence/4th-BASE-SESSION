import torch.nn as nn

from model.encoder import Encoder #인코더
from model.decoder import Decoder #디코더


class Transformer(nn.Module): #transformer 
    def __init__(self, params): #초기화
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target): #순전파
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source) #인코더의 결과값                           # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        # target: 이전 단어들의 예측에 기초해 하나씩 생성함 
        return output, attn_map # 결과값 + target 시퀀스 이전 위치에 대한 정보 포함된 attention map도 반환

    def count_params(self): # 모델의 학습 가능한 파라미터 수 계산 후 반환
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    #파라미터 개수가 많은 모델을 학습이 느려지거나 메모리가 부족해지기 때문에 대비해서 하드웨어 준비하는 등 모델 최적화하는데 유용하게 사용됨
