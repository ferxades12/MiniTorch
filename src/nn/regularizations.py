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