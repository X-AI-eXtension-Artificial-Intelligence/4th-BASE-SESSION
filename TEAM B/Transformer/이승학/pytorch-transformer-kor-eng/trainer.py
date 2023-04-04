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
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter

        self.model = Transformer(self.params)                                    #model 지정
        self.model.to(self.params.device)                                        #model을 GPU에 올림 / model(Transformer)은 model.transformer의 class Transformer에 있음

        # Scheduling Optimzer
        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),    # optimizer로 Adam(betas, eps 지정)
            hidden_dim=params.hidden_dim,                                        # params에 지정된 인자들 할당
            warm_steps=params.warm_steps
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx)   # loss로 crossEntropyLoss 사용
        self.criterion.to(self.params.device)                                    # loss도 GPU에 올림

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')                                             

        for epoch in range(self.params.num_epoch):                                 
            self.model.train()                                                   # mode 지정
            epoch_loss = 0                                                       # loss를 보기위해 epoch_loss 초기값 0, 걸린 시간을 보기위해 time.time()사용
            start_time = time.time()

            for batch in self.train_iter:                                        # zero_grad()
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()                                       
                source = batch.kor                                               # source와 target을 배치단위로 할당
                target = batch.eng

                # target sentence consists of <sos> and following tokens (except the <eos> token)
                output = self.model(source, target[:, :-1])[0]                   # model에 source를 넣은 후 나온 output

                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                output = output.contiguous().view(-1, output.shape[-1])         # output을 view로 flatten 시킴 
                target = target[:, 1:].contiguous().view(-1)                    # target도 flatten
                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]
                loss = self.criterion(output, target)                           # loss 구하기
                loss.backward()                                                 # backpropagation

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)   # gredient clip

                self.optimizer.step()                                           # 학습 반영

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item()                                       # loss 확인을 위해 +loss

            train_loss = epoch_loss / len(self.train_iter)                      # 모든 output에 대해 더한 epoch_loss를 데이터 수로 나눔
            valid_loss = self.evaluate()                                        # 

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)           # end_time 확인, 

            if valid_loss < best_valid_loss:                                    # valid_loss가 더 좋아졌다면 갱신 후 model save
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s') # 결과 print
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0                                                    

        with torch.no_grad():                                              # evaluate - 에폭 성능 확인 
            for batch in self.valid_iter:
                source = batch.kor
                target = batch.eng

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter)

    def inference(self):                                                  # model.pt를 불러와 성능을 확인
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
