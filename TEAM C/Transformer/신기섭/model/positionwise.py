import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params, pre_trained=False):
        super(PositionWiseFeedForward, self).__init__() 
        # nn.Conv1d takes input whose size is (N, C): N is a batch size, C denotes a number of channels
        self.pre_trained = pre_trained
        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1) # 배치사이즈를 받기 위해 합성곱 계층 생성
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1)
        
        if self.pre_trained:
            pass
        else:
            init_weight(self.conv1)
            init_weight(self.conv2)
        
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        # x = [batch size, sentence length, hidden dim]

        # permute x's indices to apply nn.Conv1d on input 'x'
        x = x.permute(0, 2, 1)                        # x = [batch size, hidden dim, sentence length]
                                                      # 0차원은 배치 사이즈이기 때문에 고정하고 나머지를 바꿔줌으로써 Transpose를 구현
        output = self.dropout(F.relu(self.conv1(x)))  # output = [batch size, feed forward dim, sentence length)
        output = self.conv2(output)                   # output = [batch size, hidden dim, sentence length)

        # permute again to restore output's original indices
        output = output.permute(0, 2, 1)              # output = [batch size, sentence length, hidden dim]
        return self.dropout(output)
