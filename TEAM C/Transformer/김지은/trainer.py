import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from model.optim import ScheduledAdam #adam
from model.transformer import Transformer #transformer

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        # Train mode
        if mode == 'train': #mode가 train일 경우
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else: # test mode인 경우
            self.test_iter = test_iter

        self.model = Transformer(self.params) #self.model에 transformer 모델 초기화
        self.model.to(self.params.device) #GPU로 이동

        # Scheduling Optimzer
        self.optimizer = ScheduledAdam( #adam optimizer 사용함
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
            #learning rate 조절
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_idx) #교차엔트로피loss 사용
        self.criterion.to(self.params.device)#GPU로 이동

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        for epoch in range(self.params.num_epoch): #현재 epoch에 대해 train loss초기화
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            for batch in self.train_iter: #train_iter에 대해 반복
                # For each batch, first zero the gradients
                self.optimizer.zero_grad()
                source = batch.eng
                target = batch.kor
                ######### 둘이 바꿔줌 #########

                # target sentence consists of <sos> and following tokens (except the <eos> token)
                output = self.model(source, target[:, :-1])[0] #sos를 포함하면서 end of sequence를 제외한 애들을 타겟으로
                #sos부터 eos앞의 문장까지를 output으로 받음음

                # ground truth sentence consists of tokens and <eos> token (except the <sos> token)
                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)
                #output과 target의 차원 맞추기 위해 view를 통해 변형해줌
                #output(예측값) 과 target 비교해서 손실값 계산
                
                # output = [(batch size * target length - 1), output dim]
                # target = [(batch size * target length - 1)]
                loss = self.criterion(output, target) #crossentropy 통해서 loss 진행
                loss.backward()

                # clip the gradients to prevent the model from exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip) 
                #ex) 임계값(1)이 넘는 gradient가 발견되면 모든 gradient 크기를 조정해서 벗어나지 않도록 함
                #기울기 clipping (기울기 크기 제한)

                self.optimizer.step() #optimizer을 통해 파라미터 업데이트
                

                # 'item' method is used to extract a scalar from a tensor which only contains a single value.
                epoch_loss += loss.item() #손실값 더해줌

            train_loss = epoch_loss / len(self.train_iter)
            valid_loss = self.evaluate() # epoch 끝날 때마다 train loss와 validation loss를 추력함

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model) #validation loss가 이전 최고 validation loss보다 낮으면 모델 저장

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    def evaluate(self): #검증 데이터셋에 대한 손실값 계산하는 메서드
        self.model.eval()
        epoch_loss = 0 #초기화

        with torch.no_grad(): # 평가과정에서는 모델 파라미터 업데이트를 시키지 않기 때문에 with torch.no_grad로 감싸줌
            for batch in self.valid_iter: #미니배치 단위로 모델에 입력
                source = batch.eng #####바꿈#####
                target = batch.kor #####바꿈#####

                output = self.model(source, target[:, :-1])[0]
                #마지막 eos는 제외

                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)
                #차원 맞춰주기 위해

                loss = self.criterion(output, target) #출력값과 정답값 간의 손실 계산

                epoch_loss += loss.item() # 다 더해줌

        return epoch_loss / len(self.valid_iter) # 모든 미니배치에 대한 손실값의 평균값 반환

    def inference(self): #테스트
        #모델의 저장된 가중치를 로드한걸 가지고 테스트 데이터셋에 대해 진행 -> 손실값 출력
        self.model.load_state_dict(torch.load(self.params.save_model)) #훈련된 모델의 가중치를 불러옴
        self.model.eval()
        epoch_loss = 0 #초기화

        with torch.no_grad(): #경사도 계산 비활성화(파라미터 업데이트 시키지 않음 -> 평가니까)
            for batch in self.test_iter:
                source = batch.eng #####바꿈#####
                target = batch.kor #####바꿈#####

                output = self.model(source, target[:, :-1])[0]

                output = output.contiguous().view(-1, output.shape[-1])
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()
                #최종 테스트 손실값 구함

        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f}')
