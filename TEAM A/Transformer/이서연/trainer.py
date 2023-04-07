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
            self.train_iter = train_iter ## train 데이터셋 iterator
            self.valid_iter = valid_iter ## validation 데이터셋 iterator

        # Test mode
        else:
            self.test_iter = test_iter ## test 데이터셋 iterator

        self.model = Transformer(self.params) ## Transformer 모델 생성성
        self.model.to(self.params.device) ## 파라미터 device로 전달달

        # Scheduling Optimzer
        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
        )

        ## 손실함수수
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)
        self.criterion.to(self.params.device)

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters') ## 파라미터 개수수
        best_valid_loss = float('inf') ## 가장 좋은 validation loss 초기화화

        for epoch in range(self.params.num_epoch):
            self.model.train() ## train 모드로 설정
            epoch_loss = 0 ## 현재 epoch에서의 loss 초기화
            start_time = time.time()

            for batch in self.train_iter:
                # For each batch, first zero the gradients
                self.optimizer.zero_grad() ## gradient 0으로 초기화화
                source = batch.eng #!! ## source 문장 가져옴옴
                target = batch.kor #!! ## target 문장 가져옴옴

                # target sentence consists of <sos> and following tokens (except the <eos> token)
                output = self.model(source, target[:, :-1])[0] ## output 문장 생성성

                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                output = output.contiguous().view(-1, output.shape[-1]) ## output 문장 flatten
                target = target[:, 1:].contiguous().view(-1) ## target 문장 flatten
                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]
                loss = self.criterion(output, target) ## loss 계산산
                loss.backward() ## gradient 계산산

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)

                self.optimizer.step() ## 파라미터 업데이트트

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item() ## 현재 batch에서의 loss를 epoch loss에 더해줌줌

            train_loss = epoch_loss / len(self.train_iter) ## 현재 epoch에서 train set에 대한 loss
            valid_loss = self.evaluate() ## 현재 epoch에서 validation set에 대한 loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time) ## epoch 소요시간간

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss ## 현재까지 validation set에서의 최소 loss
                torch.save(self.model.state_dict(), self.params.save_model) ## 가장 낮은 validation set loss 모델 저장장

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    def evaluate(self): ## validation set을 이용하여 모델 성능 측정
        self.model.eval() ## 평가 모드로 설정
        epoch_loss = 0 ## 초기 loss 0으로 설정

        with torch.no_grad():
            for batch in self.valid_iter:
                source = batch.eng #!! ## source 문장 가져옴
                target = batch.kor #!! ## target 문장 가져옴

                output = self.model(source, target[:, :-1])[0] ## output 문장

                output = output.contiguous().view(-1, output.shape[-1]) ## output 문장 flatten
                target = target[:, 1:].contiguous().view(-1) ## target 문장 flatten

                loss = self.criterion(output, target) ## loss 계산

                epoch_loss += loss.item() ## batch에서의 loss를 epoch loss에 더해줌

        return epoch_loss / len(self.valid_iter) ## 전체 validation set의 평균 loss 반환

    def inference(self):
        self.model.load_state_dict(torch.load(self.params.save_model)) ## 학습된 모델의 가중치 불러옴
        self.model.eval() ## evaluation 모드
        epoch_loss = 0 ## 초기 loss 0으로 설정

        with torch.no_grad(): ## gradient를 계산하지 않고 loss만 계산산 
            for batch in self.test_iter:
                source = batch.eng #!! ## source 문장 가져옴
                target = batch.kor #!! ## target 문장 가져옴

                output = self.model(source, target[:, :-1])[0] ## output 문장 생성성

                output = output.contiguous().view(-1, output.shape[-1]) ## output 문장 flatten
                target = target[:, 1:].contiguous().view(-1) ## target 문장 flatten

                loss = self.criterion(output, target) ## loss 계산

                epoch_loss += loss.item() ## epoch loss에 누적

        test_loss = epoch_loss / len(self.test_iter) ## 전체 test set에 대한 평균 loss 계산
        print(f'Test Loss: {test_loss:.3f}')
