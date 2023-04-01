import numpy as np


class ScheduledAdam(): ## 모델 최적화
    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5) ## 초기 학습률 설정
        self.optimizer = optimizer
        self.current_steps = 0 ## 현재 step의 정보를 저장하는 변수 초기화
        ## warm_steps: 학습 시 learning rate를 초기에 작게 설정해 놓은 후, 일정 기간동안 해당 값을 조절하며 학습률을 증가시키는 기법
        self.warm_steps = warm_steps ## warmup step의 수를 저장하는 변수 초기화
    
    ## 최적화 단계 수행
    def step(self): 
        # Update learning rate using current step information
        self.current_steps += 1 ## step 정보 1 증가
        lr = self.init_lr * self.get_scale() ## 학습률 계산
        
        ## optimizer 객체에 저장된 모든 파라미터의 학습률을 업데이트
        for p in self.optimizer.param_groups:
            p['lr'] = lr

        ## optimizer의 step 메소드를 호출
        self.optimizer.step()
    
    ## 모든 파라미터의 기울기를 0으로 초기화
    def zero_grad(self): 
        self.optimizer.zero_grad()

    ## 현재 step 정보와 초기 warmup step 정보를 이용하여 학습률의 scale factor를 계산
    def get_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -0.5)
        ])
