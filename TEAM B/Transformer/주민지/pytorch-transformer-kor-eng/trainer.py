import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from model.optim import ScheduledAdam
from model.transformer import Transformer

random.seed(32)
torch.manual_seed(32)
# 결정론적 알고리즘 True
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        # Train mode
        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter

        # Transformer model 생성
        self.model = Transformer(self.params) 
        self.model.to(self.params.device)

        # Scheduling Optimzer
        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
        )

        #참고) CrossEntropyLoss : LogSoftmax와 NLLLoss의 연산조합
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)
        self.criterion.to(self.params.device)

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            for batch in self.train_iter: # batch단위로 train데이터 가져오기
                # For each batch, first zero the gradients
                self.optimizer.zero_grad() # gradient 초기화
                source = batch.kor # encoder input (입력)
                target = batch.eng # decoder input (출력)

                # target sentence consists of <sos> and following tokens (except the <eos> token)
                output = self.model(source, target[:, :-1])[0]

                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                output = output.contiguous().view(-1, output.shape[-1]) # batch_first
                target = target[:, 1:].contiguous().view(-1) # 1차원으로 변경
                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]
                loss = self.criterion(output, target)
                loss.backward() # loss로 gradient 계산

                # clip the gradients to prevent the model from exploding gradient
                # gradient를 일정한 범위 내로 제한
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)

                self.optimizer.step()

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(self.train_iter)
            valid_loss = self.evaluate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # loss가 최적값보다 더 작아질 경우, 파라미터 저장(state_dict())
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0

        # no_grad에서 계산(gradient 계산 비활성화 for 학습x 결과확인o)
        with torch.no_grad():
            for batch in self.valid_iter:
                source = batch.kor
                target = batch.eng

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1]) # batch_first
                target = target[:, 1:].contiguous().view(-1) # 1차원으로 변경

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter)

    # 추론
    def inference(self):
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()
        epoch_loss = 0

        # no_grad에서 계산(gradient 계산 비활성화)
        with torch.no_grad():
            for batch in self.test_iter:
                source = batch.kor
                target = batch.eng

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1]) # batch first
                target = target[:, 1:].contiguous().view(-1) # 1차원으로 변경

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f}')
