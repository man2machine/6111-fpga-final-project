# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:27:02 2021

@author: Shahir
"""

import torch.nn as nn

class ReLUToyModel(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 layer_dims=(100, 100),
                 bias=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for next_dim in layer_dims:
            layers.append(nn.Linear(prev_dim, next_dim, bias=bias))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
