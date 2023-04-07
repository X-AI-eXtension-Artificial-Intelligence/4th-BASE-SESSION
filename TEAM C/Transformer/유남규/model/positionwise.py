import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        # 1D Convolution Layer 적용
        # input shape: (batch_size, hidden_dim, sentence_length)
        # output shape: (batch_size, feed_forward_dim, sentence_length)
        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1)
        # 두 개의 Conv1d 레이어의 weight 값들 초기화
        init_weight(self.conv1)
        init_weight(self.conv2)
        # Dropout 레이어 적용
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        # 입력 x shape: (batch_size, sentence_length, hidden_dim)
        # permute를 통해 input shape를 (batch_size, hidden_dim, sentence_length)로 변환
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, sentence_length)
        # 첫 번째 Conv1d 레이어 적용
        conv_output = self.conv1(x)  # (batch_size, feed_forward_dim, sentence_length)
        # ReLU activation 함수 적용
        activated_output = F.relu(conv_output)
        # Dropout 적용
        output = self.dropout(activated_output)
        # 두 번째 Conv1d 레이어 적용
        output = self.conv2(output)  # (batch_size, hidden_dim, sentence_length)
        # permute를 통해 output shape를 (batch_size, sentence_length, hidden_dim)으로 변환
        output = output.permute(0, 2, 1)  # (batch_size, sentence_length, hidden_dim)
        # 마지막으로 Dropout 적용
        return self.dropout(output)
