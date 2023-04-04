import numpy as np


class ScheduledAdam(): #optimizer 함수
    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer #adam optimizer 사용
        self.current_steps = 0
        self.warm_steps = warm_steps #learning rate warmup하기 위한 step 수
        # 학습 초기에서는 학습률을 낮게 설정 -> 이후 학습률을 증가시켜 나가는 것 (기울기 소실이나 증폭 문제 막는데 유용)
        
    def step(self):
        # Update learning rate using current step information
        self.current_steps += 1
        lr = self.init_lr * self.get_scale() #optimizer 업데이트하는 동시에 learning rate 조절
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr #learning rate 업데이트하는 부분

        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad() #gradient를 0으로 초기화

    def get_scale(self): #현재 step의 learning rate scale을 계산하는 메서드
        return np.min([ 
            np.power(self.current_steps, -0.5), # 현재 step에서의 learning rate scale 값을 계산함 ()
            self.current_steps * np.power(self.warm_steps, -0.5) #현재 step값이 self.warm_step보다 작을 경우, 현재 step값과 self.warm_steps 값을 곱하고 -0.5를 거듭제곱
        ]) #warmup 이후에는 learning rate를 감소시켜 안정적으로 진행
