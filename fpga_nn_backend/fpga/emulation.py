# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:17:17 2021

@author: Shahir
"""

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
"""

# =============================================================================
# Parameters
# =============================================================================

# Global params
BRAM_SIZE = 2**19 # size in bytes
DATA_BOUNDS = (-128, 127) # 8 bit signed

# mac params
NUM_MACS = 12
MAC_LANES = 8

# implementation params
BRAM_BANKS_PARAMETERS = 1
BRAM_BANKS_SCRATCHPAD = 128
BRAM_STACK_SIZE = 64

# =============================================================================
# Emulation
# =============================================================================

class FPGAEmulator:
    pass

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
        self.banks.append(bounds)
        if init_vals is not None:
            for i in range(size):
                val = init_vals[i]
                addr = i + bounds[0]
                self.write(addr, val)
    
    def write_bank(self, bank, addr, val):
        assert 0 <= bank < self.len(self.banks)
        bounds = self.banks[bank]
        assert 0 <= addr <= (bounds[1] - bounds[0])
        addr = bounds[0] + addr
        self.write(addr, val)

# =============================================================================
# Variables
# =============================================================================

bram_stack = {
    'addrs': [0]*BRAM_STACK_SIZE,
    'next': 0
}

layer_config = 0

# =============================================================================
# Iterators
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

bfs_iterator_1d = serial_iterator_1d

def bfs_iterator_2d(shape, serial_axes=(False, False)):
    pass

def bfs_iterator_3d(shape, serial_axes=(False, False, False)):
    pass

def bfs_iterator_4d(shape, serial_axes=(False, False, False, False)):
    pass

serial_iterators = {
    1: serial_iterator_1d,
    2: serial_iterator_2d,
    3: serial_iterator_3d,
    4: serial_iterator_4d
}

bfs_iterators = {
    1: bfs_iterator_1d,
    2: bfs_iterator_2d,
    3: bfs_iterator_3d,
    4: bfs_iterator_4d
}

# =============================================================================
# Initialization Utilities
# =============================================================================

def store_linear_weight(bram, weight, shape, addr, start_bank):
    assert len(shape) == 2
    O, I = shape
    b = 0
    offset = 0
    for o, i in serial_iterator_2d(shape):
        bram.bank_write(b + start_bank, addr + offset, weight[o, i])
        b = (b + 1) % BRAM_BANKS_PARAMETERS
        if b == BRAM_BANKS_PARAMETERS - 1:
            offset = offset + 1

def store_linear_bias(bram, bias, shape, addr, start_bank):
    assert len(shape) == 1
    O, = shape
    b = 0
    offset = 0
    for o, in serial_iterator_1d(shape):
        bram.bank_write(b + start_bank, addr + offset, bias[o])
        b = (b + 1) % BRAM_BANKS_PARAMETERS
        if b == BRAM_BANKS_PARAMETERS - 1:
            offset = offset + 1

def store_conv_weight(bram, weight, shape, addr, start_bank):
    assert len(shape) == 4
    O, I, H, W = shape
    b = 0
    offset = 0
    for o, i, h, w in serial_iterator_4d(shape):
        bram.bank_write(b + start_bank, addr + offset, weight[o, i, h, w])
        b = (b + 1) % BRAM_BANKS_PARAMETERS
        if b == BRAM_BANKS_PARAMETERS - 1:
            offset = offset + 1

def store_conv_bias(bram, bias, shape, addr, start_bank):
    assert len(shape) == 1
    O, = shape
    b = 0
    offset = 0
    for o, in serial_iterator_1d(shape):
        bram.bank_write(b + start_bank, addr + offset, bias[o])
        b = (b + 1) % BRAM_BANKS_PARAMETERS
        if b == BRAM_BANKS_PARAMETERS - 1:
            offset = offset + 1

# =============================================================================
# Main bram accessing interface
# =============================================================================

def load_linear_weight_value(addr, shape, o, i):
    assert len(shape) == 2

def load_linear_bias_value(addr, shape, o):
    assert len(shape) == 1

def store_linear_layer_value(addr, shape, w):
    assert len(shape) == 1
    
def load_linear_layer_value(addr, shape, w):
    assert len(shape) == 1
    
def load_conv_weight_value(addr, shape, o, i, h, w):
    assert len(shape) == 4

def load_conv_bias_value(addr, shape, o):
    assert len(shape) == 1

def store_conv_layer_value(addr, shape, c, h, w):
    assert len(shape) == 3
    
def load_conv_layer_value(addr, shape, c, h, w):
    assert len(shape) == 3

# =============================================================================
# MAC
# =============================================================================

def mac(w_in, i_in, b_in):
    o_out = [0] * MAC_LANES
    for i in range(MAC_LANES):
        o_out[i] = w_in[i] * i_in[i] + b_in[i]
    return o_out


# =============================================================================
# Serializers & Deserializer
# =============================================================================

"""
def linear_loop(
    M,
    CHW,
    input_addr,
    weight_addr,
    bias_addr,
    output_addr):
    # Shapes:
    # input: (CHW,)
    # weight: (M, CHW)
    # bias: (M,)
    # output: (M,)

    # addrs_* are arrays containing tuples of (addr in bram, bram bank)
    addrs_w = [(0, 0)]*MAC_LANES
    addrs_b = [(0, 0)]*MAC_LANES
    addrs_i = [(0, 0)]*MAC_LANES
    addrs_o = [(0, 0)]*MAC_LANES
    
    bank_w, bank_b, bank_i, bank_o = 0, 0, 0, 0

    num_level_1 = (M // MAC_LANES) + 1
    for m1 in range(num_level_1):
        for chw in range(CHW):
            # parallel loop for MAC - start

            # input
            for m0, addr in enumerate(range(chw, chw + MAC_LANES)):
                addrs_i[m0] = (addr + input_addr, bank_i)
            bank_i = (bank_i + 1) % BRAM_BANKS_SCRATCHPAD

            m1L = m1*MAC_LANES
            for m0, addr in enumerate(m1L, m1L + MAC_LANES):
                # output
                addrs_o[m0] = (addr + output_addr, bank_o)
                bank_o = (bank_o + 1) % BRAM_BANKS_SCRATCHPAD

                # bias
                addrs_b[m0] = (addr + bias_addr, bank_b)
                bank_b = (bank_b + 1) % BRAM_BANKS_PARAMETERS

            # weight
            CHW_m1L_m0 = m1L
            for m0 in range(MAC_LANES):
                addrs_w[m0] = (CHW_m1L_m0 + chw + weight_addr, bank_w)
                bank_w = (bank_w + CHW) % BRAM_BANKS_PARAMETERS
                CHW_m1L_m0 += CHW
            
            # parallel loop for MAC - end
            
            bank_w = (bank_w + 1) % BRAM_BANKS_PARAMETERS

            yield (addrs_i, addrs_w, addrs_b, addrs_o)
"""

def linear_loop2(
    M,
    CHW,
    input_addr,
    weight_addr,
    bias_addr,
    output_addr):
    # Shapes:
    # input: (CHW,)
    # weight: (M, CHW)
    # bias: (M,)
    # output: (M,)
    
    # addrs_* are arrays containing tuples of (addr in bram, bram bank)
    addrs_w = [(0, 0)]*MAC_LANES
    addrs_b = [(0, 0)]*MAC_LANES
    addrs_i = [(0, 0)]*MAC_LANES
    addrs_o = [(0, 0)]*MAC_LANES

    bank_w, bank_b, bank_i, bank_o = 0, 0, 0, 0

    num_level_1 = (M // MAC_LANES) + 1
    for m1 in range(num_level_1):
        for chw in range(CHW):
            # parallel loop for MAC - start

            m1L = m1*MAC_LANES
            CHW_m1L_m0 = m1*MAC_LANES
            for m0 in range(MAC_LANES):
                # input
                addrs_i[m0] = (chw + input_addr, bank_i)
                
                # output
                addrs_o[m0] = (m1L + m0 + output_addr, bank_o)
                bank_o = (bank_o + 1) % BRAM_BANKS_SCRATCHPAD

                # bias
                addrs_b[m0] = (m1L + m0 + bias_addr, bank_b)
                bank_b = (bank_b + 1) % BRAM_BANKS_PARAMETERS

                # weight
                addrs_w[m0] = (CHW_m1L_m0 + chw + weight_addr, bank_w)
                bank_w = (bank_w + CHW) % BRAM_BANKS_PARAMETERS
                CHW_m1L_m0 += CHW
                
            yield (addrs_i, addrs_w, addrs_b, addrs_o)
            
            # parallel loop for MAC - end
            
            bank_i = (bank_i + 1) % BRAM_BANKS_SCRATCHPAD
            bank_w = (bank_w + 1) % BRAM_BANKS_PARAMETERS

"""
def linear_loop3(
    M,
    CHW,
    input_addr,
    weight_addr,
    bias_addr,
    output_addr):
    # Shapes:
    # input: (CHW,)
    # weight: (M, CHW)
    # bias: (M,)
    # output: (M,)

    # addrs_* are tuples of (addr in bram, bram bank)
    
    bank_w, bank_b, bank_i, bank_o = 0, 0, 0, 0

    for m in range(M):
        # output
        addrs_o = (m + output_addr, bank_o)
        bank_o = (bank_o + 1) % BRAM_BANKS_SCRATCHPAD

        # bias
        addrs_b = (m + bias_addr, bank_b)
        bank_b = (bank_b + 1) % BRAM_BANKS_PARAMETERS

        CHWm = 0
        for chw in range(CHW):
            # input
            addrs_i = (chw + input_addr, bank_i)
            bank_i = (bank_i + 1) % BRAM_BANKS_SCRATCHPAD

            # weight
            addrs_w = (CHWm + chw + weight_addr, bank_w)
            bank_w = (bank_w + 1) % BRAM_BANKS_PARAMETERS
            
            yield (addrs_i, addrs_w, addrs_b, addrs_o)            
         
        bank_w = (bank_w + CHW) % BRAM_BANKS_PARAMETERS
        CHWm += CHW
"""

def conv_loop():
    pass

def sum_loop():
    pass

def flatten_loop():
    pass

# =============================================================================
# BRAM Stack
# =============================================================================

def bram_stack_add(alloc_size):
    # returns allocation start addr
    if bram_stack['next'] == 0:
        last_addr = 0
    else:
        last_addr = bram_stack['addrs'][bram_stack['next'] - 1]
    bram_stack['addrs'][bram_stack['next']] = last_addr + alloc_size
    bram_stack['next'] += 1

def bram_stack_top_remove():
    if bram_stack['next'] > 0:
        bram_stack['next'] -= 1

def bram_stack_top_move(end_index):
    assert end_index < bram_stack['next']
    pass

# =============================================================================
# Overall FSM
# =============================================================================

def execute_layer(layer_type,
                  layer_input_shape,
                  layer_output_shape,
                  stack_input_index,
                  stack_output_index):
    pass

def overall_fsm():
    pass
