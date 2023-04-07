import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        # nn.Conv1d takes input whose size is (N, C): N is a batch size, C denotes a number of channels
        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1) ## 합성곱 층 초기화
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1)
        init_weight(self.conv1) ## 가중치 초기화
        init_weight(self.conv2)
        self.dropout = nn.Dropout(params.dropout) ## 드롭아웃 초기화화

    def forward(self, x):
        # x = [batch size, sentence length, hidden dim]

        # permute x's indices to apply nn.Conv1d on input 'x'
        x = x.permute(0, 2, 1) ## 차원 순서 바꿔줌 (batch size, hidden dim, sentence length) -> 1차원 합성공 적용                      # x = [batch size, hidden dim, sentence length]
        output = self.dropout(F.relu(self.conv1(x))) ## conv1, 활성화함수 적용  # output = [batch size, feed forward dim, sentence length)
        output = self.conv2(output)                   # output = [batch size, hidden dim, sentence length)

        # permute again to restore output's original indices
        output = output.permute(0, 2, 1) ## 원래의 shape로 복원원            # output = [batch size, sentence length, hidden dim]
        return self.dropout(output)
