from src.tensor import Tensor
import numpy as np

class Module:
    def __init__(self):
        """Base class for all neural networks and layers.
        """
        self.submodules = []
        self.params = []
        self.training: bool = True
        self.weight = []

    def __call__(self, *args):
        return self.forward(*args)
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if isinstance(value, Module):
            self.submodules.append(value)

    def forward(self):
        raise  NotImplementedError

    def parameters(self) -> list:
        """Return the list of all parameters in the module and its submodules.
        """
        params = self.params

        for module in self.submodules:
            params.extend(module.parameters())

        return params

    def get_weights(self) -> list[Tensor]:
        weights = []

        for module in self.submodules:
            weights.extend(module.weight)

        return weights
    
    def eval(self) -> None:
        self.training = False

        for module in self.submodules:
            module.training = False
    
    def train(self) -> None:
        self.training = True

        for module in self.submodules:
            module.training = True
    
    def predict(self, x:Tensor) -> Tensor:
        eval = self.training

        self.eval()


        preds = self.forward(x).data

        if preds.ndim > 1 and preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
        else:
            preds = (preds > 0.5).astype(int)
        
        if eval:
            self.training()
            
        return Tensor(preds)

    