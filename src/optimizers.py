import numpy as np

class Optimizer():
    """Base class for all optimizers.
    """
    def __init__(self, parameters, lr : float = 0.001):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        """Reset gradients of all parameters.
        """
        for param in self.parameters:
            param.grad = None

class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum :float = 0, dampening: float = 0, maximize: bool = False):
        """Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters (Iterable[Parameter]): The parameters to optimize.
            lr (float): The learning rate.
            momentum (float, optional): Momentum factor. Defaults to 0.
            dampening (float, optional): Dampening for momentum. Defaults to 0.
            maximize (bool, optional): Whether to maximize the objective. Defaults to False.
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.maximize = maximize
        self.updates = [np.zeros_like(p.data) for p in parameters]

        if momentum == 0: self.dampening = 0

    def step(self):
        """Perform a single optimization step.

        General formula for SGD with or without momentum:
        upd_t = momentum * upd_(t-1) + (1 - dampening) * g
        param = param - lr * upd  (if minimizing)
        """
        for i, param in enumerate(self.parameters):
            if param.grad is None: continue

            g = param.grad
            self.updates[i] = self.momentum * self.updates[i] + (1 - self.dampening) * g
            
            if self.maximize:
                param.data = param.data + self.lr * self.updates[i]
            else:
                param.data = param.data - self.lr * self.updates[i]
    
    

class Adam(Optimizer):
    def __init__(self, parameters, lr = 0.001, betas=(0.9, 0.999)):
        """Adam optimizer.

        Creates m and v vectors for each parameter, with the same shape as the parameter.

        Args:
            parameters (Iterable[Parameter]): The parameters to optimize.
            lr (float, optional): The learning rate. Defaults to 0.001.
            betas (tuple, optional): Coefficients for computing running averages of gradient and its square. Defaults to (0.9, 0.999).

        Raises:
            ValueError: If the betas tuple is not of length 2.
        """
        super().__init__(parameters, lr)

        if len(betas) != 2:
            raise ValueError("Wrong betas size")
        
        self.b1 = betas[0]
        self.b2 = betas[1]

        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

        self.iterations = 0

    def step(self):
        """Perform a single optimization step.

        Computes biased first and second moment estimates, corrects the bias, and updates parameters.
        """
        self.iterations += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None: continue

            g = param.grad

            # First and second moment estimates
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2

            # Correcting the bias
            m_unbiased = self.m[i] / (1 - self.b1 ** self.iterations)
            v_unbiased = self.v[i] / (1 - self.b2 ** self.iterations)

            # Update parameters
            value = param.data - self.lr * m_unbiased / (np.sqrt(v_unbiased) + 1e-8)

            param.data = value
