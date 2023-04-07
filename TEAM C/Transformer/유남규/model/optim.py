import numpy as np


class ScheduledAdam():
    def __init__(self, optimizer, hidden_dim, warm_steps):
        # 초기 학습률: sqrt(hidden_dim) 의 제곱근 역수
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # 현재 스텝 수 증가
        self.current_steps += 1
        # 현재 스텝 수에 따른 학습률 계산
        lr = self.init_lr * self.get_scale()

        # optimizer의 각 파라미터 그룹에 대해 학습률 업데이트
        for p in self.optimizer.param_groups:
            p['lr'] = lr

        # optimizer의 step 메소드 호출
        self.optimizer.step()

    def zero_grad(self):
        # optimizer의 zero_grad 메소드 호출
        self.optimizer.zero_grad()

    def get_scale(self):
        # 현재 스텝 수와 warm-up 스텝 수를 이용하여 스케일 계산
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -0.5)
        ])
