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
NUM_MACS = 12
MAC_LANES = 8

# implementation params
BRAM_BANKS_PARAMETERS = 1 # prime number
BRAM_BANKS_SCRATCHPAD = 101 # prime number
BRAM_STACK_SIZE = 64

class LayerType(Enum):
    DENSE = 0
    CONV = 1
    RELU = 2
    RELU6 = 3
    SUM = 4
    FLATTEN = 5

class IterationStrategy(Enum):
    SERIAL = 0
    ADVANCED = 1

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

class BRAMStack:
    def __init__(self, stack_size=16):
        self.stack_addrs = [0] * (stack_size + 1)
        self.stack_next = 0
        self.stack_size = stack_size
    
    def add(self, alloc_size):
        # returns allocation start addr
        if self.stack_next == 0:
            last_addr = 0
        else:
            last_addr = self.stack_addrs[self.stack_next - 1]
        self.stack_addrs[self.stack_next] = last_addr + alloc_size
        self.stack_next += 1

    def bram_stack_top_remove(self):
        if self.stack_next > 0:
            self.stack_next -= 1

    def bram_stack_top_move(self, end_index):
        assert end_index < self.stack_next
        pass

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
# MAC
# =============================================================================

def mac(w_in, i_in, b_in, mac_lanes=MAC_LANES):
    o_out = [0] * mac_lanes
    for i in range(mac_lanes):
        o_out[i] = w_in[i] * i_in[i] + b_in[i]
    return o_out

# =============================================================================
# Serializers & Deserializer
# =============================================================================

def linear_loop(
    M,
    CHW,
    M1,
    input_addr,
    weight_addr,
    bias_addr,
    output_addr,
    mac_lanes=MAC_LANES):
    # Shapes:
    # input: (CHW,)
    # weight: (M, CHW)
    # bias: (M,)
    # output: (M,)
    
    # addrs_* are arrays containing tuples of (addr in bram, bram bank)
    addrs_w = [(0, 0)]*mac_lanes
    addrs_b = [(0, 0)]*mac_lanes
    addrs_i = [(0, 0)]*mac_lanes
    addrs_o = [(0, 0)]*mac_lanes

    bank_w, bank_b, bank_i, bank_o = 0, 0, 0, 0

    assert mac_lanes < BRAM_BANKS_SCRATCHPAD
    assert M1 == math.ceil(M / mac_lanes)
    CHW_M0 = CHW * MAC_LANES
    m1L = 0
    for m1 in range(M1):
        for chw in range(CHW):
            # parallel loop for MAC - start

            # m1L = m1 * mac_lanes
            CHW_m1L_m0 = m1L
            for m0 in range(mac_lanes):
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
            
            bank_o = (bank_o - mac_lanes) % BRAM_BANKS_SCRATCHPAD
            bank_b = (bank_b - mac_lanes) % BRAM_BANKS_SCRATCHPAD
            bank_w = (bank_w - CHW_M0) % BRAM_BANKS_SCRATCHPAD
            
            yield (addrs_i, addrs_w, addrs_b, addrs_o)
            
            # parallel loop for MAC - end
            
            bank_i = (bank_i + 1) % BRAM_BANKS_SCRATCHPAD
            bank_w = (bank_w + 1) % BRAM_BANKS_PARAMETERS
        
        bank_o = (bank_o + mac_lanes) % BRAM_BANKS_SCRATCHPAD
        bank_b = (bank_b + mac_lanes) % BRAM_BANKS_PARAMETERS
        m1L += mac_lanes
        
            


def linear_activation_loop(
    M,
    CHW,
    M1,
    input_addr,
    output_addr,
    act_lanes=MAC_LANES):
    # Shapes:
    # input: (CHW,)
    # output: (M,)
    
    # addrs_* are arrays containing tuples of (addr in bram, bram bank)
    addrs_i = [(0, 0)]*act_lanes
    addrs_o = [(0, 0)]*act_lanes

    bank_i, bank_o = 0, 0

    assert M1 == math.ceil(M // act_lanes)
    for m1 in range(M1):
        for chw in range(CHW):
            # parallel loop for activation unit - start

            m1L = m1*act_lanes
            for m0 in range(act_lanes):
                # input
                addrs_i[m0] = (chw + input_addr, bank_i)
                
                # output
                addrs_o[m0] = (m1L + m0 + output_addr, bank_o)
                bank_o = (bank_o + 1) % BRAM_BANKS_SCRATCHPAD
                
            yield (addrs_i, addrs_o)
            
            # parallel loop for activation unit - end
            
            bank_i = (bank_i + 1) % BRAM_BANKS_SCRATCHPAD

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
    

