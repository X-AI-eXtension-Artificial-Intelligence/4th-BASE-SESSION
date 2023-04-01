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
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        # Train mode
        if mode == 'train':
            self.train_iter = train_iter # 받아옴
            self.valid_iter = valid_iter # 받아옴

        # Test mode
        else:
            self.test_iter = test_iter

        self.model = Transformer(self.params) # Transformer 지정
        self.model.to(self.params.device) # device 지정

        # Scheduling Optimzer -> optimizer 지정
        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)
        self.criterion.to(self.params.device)

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch): # -> eppoch
            self.model.train()
            epoch_loss = 0 # eppoch loss 0으로 설정
            start_time = time.time() # 시간 설정 

            for batch in self.train_iter:
                # For each batch, first zero the gradients
                # 각 배치에 대해 먼저 그라데이션을 0으로 설정합니다
                self.optimizer.zero_grad()
                source = batch.kor
                target = batch.eng

                # target sentence consists of <sos> and following tokens (except the <eos> token)
                # 대상 문장은 <target> 및 후속 토큰으로 구성됩니다(<target> 토큰 제외)
                output = self.model(source, target[:, :-1])[0]

                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                # ground truth 문장은 토큰과 <eos> 토큰으로 구성됩니다(<sos> 토큰 제외)
                output = output.contiguous().view(-1, output.shape[-1]) # contiguous 메모리 저장
                target = target[:, 1:].contiguous().view(-1)
                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]
                loss = self.criterion(output, target) # loss 계산
                loss.backward() # 역전파

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)
                # 규제 https://sanghyu.tistory.com/87

                self.optimizer.step() # 가중치 업데이트

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item() # epoch loss 

            train_loss = epoch_loss / len(self.train_iter) # train loss
            valid_loss = self.evaluate() # 함수 이동 평가

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time) # 에폭 시간

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0 # 에폭 로스 0으로 설정

        with torch.no_grad():
            for batch in self.valid_iter: 
                source = batch.kor
                target = batch.eng

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter)

    def inference(self):
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.test_iter:
                source = batch.kor
                target = batch.eng

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f}')
