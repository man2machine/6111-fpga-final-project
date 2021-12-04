# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:17:17 2021

@author: Shahir
"""

import math
from enum import Enum

import numpy as np

"""
References:
https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

Based on 6.825
Lecture 5 slide 14, 25, 32
Lecture 6 slide 8, 32, 34
Lecture 7 slide 14
Lecture 8 slide 23

Each class or top level funciton represents a module,
function is preferred and classes used only when needed
"""

# =============================================================================
# Parameters
# =============================================================================

# Global params
BRAM_SIZE = 2**19 # size in bytes
DATA_BOUNDS = (-128, 127) # 8 bit signed

# mac params
MAC_LANES = 3
NUM_MACS = 1

# implementation params
BRAM_STACK_SIZE = 64

class LayerType(Enum):
    DENSE = 0
    CONV = 1
    RELU = 2
    RELU6 = 3
    SUM = 4
    FLATTEN = 5
    MOVE = 6
    INPUT = -1 # not actually used on the FPGA

class IterationStrategy(Enum):
    SERIAL = 0

# =============================================================================
# Emulation
# =============================================================================

class BRAMEmulator:
    def __init__(self, size):
        self.size = size
        self.bram = {}
        self.banks = []
        self.last_bank_end = 0
    
    def read(self, addr):
        return self.bram.get(addr, 0)
    
    def write(self, addr, val):
        assert DATA_BOUNDS[0] <= val <= DATA_BOUNDS[1]
        self.bram[addr] = val
    
    def allocate_bank(self, size, init_vals=None):
        if init_vals is not None:
            assert len(init_vals) == size
        bounds = (self.last_bank_end, self.last_bank_end + size - 1)
        self.last_bank_end = bounds[1] + 1
        bank_index = len(self.banks)
        self.banks.append(bounds)
        if init_vals is not None:
            for i in range(size):
                val = init_vals[i]
                addr = i + bounds[0]
                self.write(addr, val)
        
        return bank_index
    
    def read_bank(self, bank, addr):
        assert 0 <= bank < self.len(self.banks)
        bounds = self.banks[bank]
        assert 0 <= addr <= (bounds[1] - bounds[0])
        addr = bounds[0] + addr

        return self.read(addr)
    
    def write_bank(self, bank, addr, val):
        assert 0 <= bank < self.len(self.banks)
        bounds = self.banks[bank]
        assert 0 <= addr <= (bounds[1] - bounds[0])
        addr = bounds[0] + addr

        return self.write(addr, val)
    
    def rw_cycle_bank(self, read_addrs, write_addrs):
        assert len(read_addrs) == len(write_addrs) == len(self.banks)

        read_outs = [0] * len(self.banks)
        for i in range(len(self.banks)):
            read_outs[i] = self.read_bank(read_addrs[i])
            self.write_bank(write_addrs[i])
        
        return read_outs

# =============================================================================
# Base weight iterators
# =============================================================================

def serial_iterator_1d(shape):
    assert len(shape) == 1
    I, = shape
    for i in range(I):
        yield (i,)

def serial_iterator_2d(shape):
    assert len(shape) == 2
    I, J = shape
    for i in range(I):
        for j in range(J):
            yield (i, j)

def serial_iterator_3d(shape):
    assert len(shape) == 3
    I, J, K = shape
    for i in range(I):
        for j in range(J):
            for k in range(K):
                yield (i, j, k)

def serial_iterator_4d(shape):
    assert len(shape) == 4
    I, J, K, L = shape
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    yield (i, j, k, l)

serial_iterators = {
    1: serial_iterator_1d,
    2: serial_iterator_2d,
    3: serial_iterator_3d,
    4: serial_iterator_4d
}

# =============================================================================
# MAC and Activation
# =============================================================================

def mac(w_in, i_in, b_in, lanes=MAC_LANES):
    o_out = [0] * lanes
    for i in range(lanes):
        o_out[i] = w_in[i] * i_in[i] + b_in[i]
    return o_out

def relu(o_in, lanes=MAC_LANES):
    a_out = [0] * lanes
    for i in range(lanes):
        if o_in < 0:
            a_out[i] = 0
        else:
            a_out[i] = o_in[i]
    return a_out

# =============================================================================
# Serializers & Deserializer Loops
# =============================================================================

def linear_layer_init_loop(
    M,
    CHW,
    bias_addr,
    output_addr):

    for m in range(M):
        # output
        addr_o = m + output_addr
        # bias
        addr_b = m + bias_addr

        yield (addr_b, addr_o)

        # o[m] = b[m]

def linear_layer_mac_loop(
    M,
    CHW,
    input_addr,
    weight_addr,
    output_addr):
    # Shapes:
    # input: (CHW,)
    # weight: (M, CHW)
    # bias: (M,)
    # output: (M,)

    for m in range(M):
        # output
        addr_o = m + output_addr

        CHWm = 0
        for chw in range(CHW):
            # input
            addr_i = chw + input_addr
            # weight
            addr_w = CHWm + chw + weight_addr
            
            yield (addr_i, addr_w, addr_o)

            # o[m] = o[m] + i[chw] * w[CHW*m + chw];
        
        CHWm += CHW

def linear_activation_loop(
    M,
    CHW,
    output_addr):
    # Shapes:
    # output: (M,)
    
    for m in range(M):
        # output
        addr_o = m + output_addr
            
        yield (addr_o,)

        # o[m] = activation(o[m])    

def conv_loop():
    pass

def conv_activation_loop():
    pass

def sum_loop():
    pass

def flatten_loop():
    pass

# =============================================================================
# Overall FSM
# =============================================================================

class Strategy1Executor:
    def __init__(self):
        pass

    def prepare(self, converted_nn):
        pass

    def execute(self):
        use_mac = False
        for _ in range(0):
            if layer_type == LayerType.DENSE:
                pass
            elif layer_type == LayerType.FLATTEN:
                pass

