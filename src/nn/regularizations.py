import numpy as np
import src.nn as nn
from src.operations import DropoutOp


class Dropout(nn.Module):
    def __init__(self, p:float = 0.5):
        super().__init__()
        self.p = p
        self.training = True
    
    def forward(self, x):
        return x._apply_unary_op(DropoutOp, self.p, self.training)
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True

class L1():
    def __init__(self, l:float = 1e-4):
        self.l = l

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, loss:float, model):
        l1_loss = loss
        for weight in model.get_weights():
            l1_loss += self.l * weight.abs().sum()
        
        return l1_loss