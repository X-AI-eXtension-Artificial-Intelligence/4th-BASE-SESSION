import numpy as np


class ScheduledAdam():
    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.current_steps = 0
        #warmup : 초기에 작은 lr을 사용하고 training과정이 안정되면 초기 lr로 전환
        self.warm_steps = warm_steps

    def step(self):
        # Update learning rate using current step information
        self.current_steps += 1
        # 계산된 scale값을 곱해주어 lr 업데이트
        lr = self.init_lr * self.get_scale()
        
        for p in self.optimizer.param_groups:
            p['lr'] = lr

        self.optimizer.step()

    # gradient를 0으로 초기화    
    def zero_grad(self):
        self.optimizer.zero_grad()

    # step수에 따라서 scale값 계산
    def get_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            self.current_steps * np.power(self.warm_steps, -0.5)
        ])
