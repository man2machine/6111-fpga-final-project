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
"""

# =============================================================================
# Parameters
# =============================================================================

DATA_BOUNDS = (-128, 127) # 8 bit signed

MAC_TOTAL_SIZE = 72
MAC_WAYS = 8
MAC_INNER_SUM_SIZE = MAC_TOTAL_SIZE / MAC_WAYS

BRAM_SIZE = 2**19 # size in bytes
BRAM_BANKS = 128
BRAM_BANKS_LOG2 = 7
BRAM_BANK_SIZE = int(BRAM_SIZE / BRAM_BANKS)

BRAM_STACK_SIZE = 64

class FPGAEmulator:
    pass

# =============================================================================
# Emulation
# =============================================================================

bram = {}

# =============================================================================
# Variables
# =============================================================================

bram_stack = {
    'addrs': [0]*BRAM_STACK_SIZE,
    'next': 0
    }

layer_config = 0

# =============================================================================
# BRAM
# =============================================================================

def bram_read(addr):
    return bram.get(addr, 0)

def bram_write(addr, val):
    assert DATA_BOUNDS[0] <= val <= DATA_BOUNDS[1]
    bram[addr] = val

def bram_bank_read(bank, addr):
    assert 0 <= bank < BRAM_BANKS
    addr = addr + bank * BRAM_BANK_SIZE
    return bram_read(addr)

def bram_bank_write(bank, addr, val):
    assert 0 <= bank < BRAM_BANKS
    addr = addr + bank * BRAM_BANK_SIZE
    return bram_write(addr, val)

def compute_per_bram_bank(total_size):
    return (total_size >> BRAM_BANKS_LOG2) + 1

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

def store_linear_weight(weight, shape, addr):
    assert len(shape) == 2
    O, I = shape
    b = 0
    offset = 0
    for o, i in serial_iterator_2d(shape):
        bram_bank_write(b, addr + offset, weight[o, i])
        b = (b + 1) % BRAM_BANKS
        if b == BRAM_BANKS - 1:
            offset = offset + 1

def store_linear_bias(bias, shape, addr):
    assert len(shape) == 1
    O, = shape
    b = 0
    offset = 0
    for o, in serial_iterator_1d(shape):
        bram_bank_write(b, addr + offset, bias[o])
        b = (b + 1) % BRAM_BANKS
        if b == BRAM_BANKS - 1:
            offset = offset + 1

def store_conv_weight(weight, shape, addr):
    assert len(shape) == 4
    O, I, H, W = shape
    b = 0
    offset = 0
    for o, i, h, w in serial_iterator_4d(shape):
        bram_bank_write(b, addr + offset, weight[o, i, h, w])
        b = (b + 1) % BRAM_BANKS
        if b == BRAM_BANKS - 1:
            offset = offset + 1

def store_conv_bias(bias, shape, addr):
    assert len(shape) == 1
    O, = shape
    b = 0
    offset = 0
    for o, in serial_iterator_1d(shape):
        bram_bank_write(b, addr + offset, bias[o])
        b = (b + 1) % BRAM_BANKS
        if b == BRAM_BANKS - 1:
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

def mac_product():
    pass

def mac_sum_reduce():
    pass

def mac():
    pass

# =============================================================================
# Serializers & Deserializer
# =============================================================================

def linear_serialize():
    pass

def linear_deserialize():
    pass

def conv_serialize():
    pass

def conv_deserialize():
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
