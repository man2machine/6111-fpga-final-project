# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:59:36 2021

@author: Shahir
"""

import torch
import torch.nn as nn

class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model # model_fp32
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
