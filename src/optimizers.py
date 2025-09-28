import numpy as np

class Optimizer():
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class GradientDescent(Optimizer):
    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data = param.data - self.lr * param.grad
    
    

class Adam(Optimizer):
    def __init__(self, parameters, lr = 0.001, betas=(0.9, 0.999)):
        super().__init__(parameters, lr)

        if len(betas) != 2:
            raise ValueError("Wrong betas size")
        
        self.b1 = betas[0]
        self.b2 = betas[1]

        self.m = [0 for _ in parameters]
        self.v = [0 for _ in parameters]

        self.iterations = 0

    def step(self):
        self.iterations += 1

        for i in range(len(self.parameters)):
            g = self.parameters[i].grad

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2

            m_unbiased = self.m[i] / (1 - self.b1 ** self.iterations)
            v_unbiased = self.v[i] / (1 - self.b2 ** self.iterations)

            value = self.parameters[i].data - self.lr * m_unbiased / (np.sqrt(v_unbiased) + 1e-8)

            self.parameters[i].data = value
