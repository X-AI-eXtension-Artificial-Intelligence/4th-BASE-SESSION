import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        # nn.Conv1d takes input whose size is (N, C): N is a batch size, C denotes a number of channels
        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1) # conv1 설정
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1) # conv2 설정
        init_weight(self.conv1) # 가중치 초기화
        init_weight(self.conv2) # 가중치 초기화
        self.dropout = nn.Dropout(params.dropout) # 드롭 아웃

    def forward(self, x):
        # x = [batch size, sentence length, hidden dim]

        # permute x's indices to apply nn.Conv1d on input 'x'
        # nn을 적용할 permute x의 인덱스.입력 'x'의 Conv1d
        x = x.permute(0, 2, 1)                        # x = [batch size, hidden dim, sentence length] #구조 변경
        output = self.dropout(F.relu(self.conv1(x)))  # output = [batch size, feed forward dim, sentence length)
        output = self.conv2(output)                   # output = [batch size, hidden dim, sentence length)

        # permute again to restore output's original indices
        output = output.permute(0, 2, 1)              # output = [batch size, sentence length, hidden dim] # 다시 구조 변경
        return self.dropout(output)
