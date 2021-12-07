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
BRAM_SIZE = 607500 # size in bytes
DATA_BOUNDS = (-128, 127) # 8 bit signed

# mac params
MAC_LANES = 1

# implementation params
BRAM_STACK_SIZE = 64

class LayerType(Enum):
    DENSE = 0
    # CONV = 1
    RELU = 2
    # RELU6 = 3
    # SUM = 4
    MOVE = 5
    OUTPUT = 6

class LayerStep(Enum):
    INIT_BIAS = 0
    MAC_OUTPUT = 1

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
    
    def rw_cycle_bank(self, bank, read_addr, write_addr):
        read_out = self.read_bank(bank, read_addr)
        self.write_bank(bank, write_addr)
        
        return read_out
    
    def rw_cycle_banks(self, read_addrs, write_addrs):
        assert len(read_addrs) == len(write_addrs) == len(self.banks)

        read_outs = [0] * len(self.banks)
        for i in range(len(self.banks)):
            read_outs[i] = self.read_bank(i, read_addrs[i])
            self.write_bank(i, write_addrs[i])
        
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

def mac(w_in, i_in, b_in, lanes=MAC_LANES, cycle_delay=3):
    o_out = [0] * lanes
    for i in range(lanes):
        o_out[i] = w_in[i] * i_in[i] + b_in[i]
    
    for c in range(cycle_delay - 1):
        yield None
    yield o_out

def relu(o_in, lanes=MAC_LANES, cycle_delay=1):
    a_out = [0] * lanes
    for i in range(lanes):
        if o_in < 0:
            a_out[i] = 0
        else:
            a_out[i] = o_in[i]
    for c in range(cycle_delay - 1):
        yield None
    yield a_out

# =============================================================================
# Serializers & Deserializer Loops
# =============================================================================

def linear_layer_init_loop(
    M,
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

    CHWm = 0
    for m in range(M):
        # output
        addr_o = m + output_addr
        
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

def move_loop(
    N,
    input_addr,
    output_addr):
    # Shapes:
    # output: (N,)
    
    for n in range(N):
        # input
        addr_i = n + input_addr
        # output
        addr_o = n + output_addr
            
        yield (addr_i, addr_o,)

        # o[n] = i[n]

# =============================================================================
# Overall FSM
# =============================================================================

class FPGAEmulator:
    def __init__(self, converted_nn):
        self.converted_nn = converted_nn
        self.bram = BRAMEmulator(BRAM_SIZE)
        weight_data = np.array(list(b''.join(self.converted_nn.generate_parameter_data())), dtype=np.int8)
        scratchpad_size = BRAM_SIZE - len(weight_data) - 1024
        
        self.preparation_info = {
            'bram_parameters_size': len(weight_data),
            'bram_scratchpad_size': scratchpad_size,
            'weight_data': weight_data
        }

        self.bram.allocate_bank(len(self.preparation_info['weight_data']),
            init_vals=self.preparation_info['weight_data']) # bank 0
        self.bram.allocate_bank(self.preparation_info['bram_scratchpad_size']) # bank 1

        self.exec_info = self.converted_nn.generate_execution_info(
            self.preparation_info['bram_scratchpad_size'])

    def execute(self, input_data):
        # EMULATION OF RETRIEVING INPUT DATA FROM EXTERNAL SOURCE =============
        # this is not meant to be translated to verilog
        input_data = input_data.reshape(self.exec_info['input_shape']).astype(np.int8)
        iterator = serial_iterators[input_data.ndim](input_data.shape)
        i = 0
        for ind in iterator:
            self.bram.write_bank(1, i + self.exec_info['inital_input_addr'], input_data[i])
            i += 1  
        # =====================================================================

        # overall layer state variables
        layer_num = 0
        layer_type = 0

        # addr bases
        input_base_addr = 0
        weight_base_addr = 0
        bias_base_addr = 0
        output_base_addr = 0

        # sizes
        m_size = 0
        chw_size = 0
        n_size = 0

        # linear layer overall state variables
        linear_layer_step = LayerStep.INIT_BIAS
        linear_has_bias = 0

        # linear init loop module signals
        linear_init_loop_bias_addr = 0
        linear_init_loop_output_addr = 0
        linear_init_loop_output_prev_addr = 0
        linear_init_loop_started = 0
        linear_init_loop_ready_out = 0
        linear_init_loop_done_out = 0
        linear_init_loop_start_ready_in = 0
        linear_init_loop_next_ready_in = 0

        # linear mac loop module signals
        linear_mac_loop_input_addr = 0
        linear_mac_loop_weight_addr = 0
        linear_mac_loop_output_addr = 0
        linear_mac_loop_output_prev_addr = 0
        linear_mac_loop_started = 0
        linear_mac_loop_ready_out = 0
        linear_mac_loop_done_out = 0
        linear_mac_loop_start_ready_in = 0
        linear_mac_loop_next_ready_in = 0
        linear_mac_loop_read_step = 0
        linear_mac_loop_read_lane_index = 0
        linear_mac_loop_write_step = 0
        linear_mac_loop_write_lane_index = 0
        linear_mac_loop_output_addrs = [0] * MAC_LANES

        # layer signals
        next_layer = 0

        # loop modules
        linear_init_loop_inst = None
        linear_mac_loop_inst = None
        move_loop_inst = None
        mac_inst = None
        relu_inst = None

        # mac module signals
        mac_ready_in = 0
        mac_done_out = 0
        weights_mac = [0] * MAC_LANES
        inputs_mac = [0] * MAC_LANES
        biases_mac = [0] * MAC_LANES
        outputs_mac = [0] * MAC_LANES
        
        # relu signal
        relu_ready_in = 0
        relu_done_out = 0
        inputs_relu = [0] * MAC_LANES
        outputs_relu = [0] * MAC_LANES
        mac_output_prev_addrs = [0] * MAC_LANES

        # bram scratchpad
        bram0_read_addr = 0
        bram0_read_enable = 0
        bram0_read_out = 0

        # bram scratchpad
        bram1_read_enable = 0
        bram1_read_addr = 0
        bram1_read_out = 0
        bram1_write_enable = 0
        bram1_write_addr = 0
        bram1_write_val = 0
        
        # mac lane idx
        mac_lane_idx = 0
        output_flag = 0

        # configuration load
        layers = self.exec_info['layers']

        # fsm loop
        while True:
            # modules emulation

            # MAC Module
            if mac_ready_in:
                mac_inst = mac(weights_mac, inputs_mac, biases_mac)
                loop_out = next(mac_inst, None)
                if loop_out is not None:
                    outputs_mac = loop_out
                    mac_done_out = 0
                else:
                    mac_done_out = 1
            
            # ReLU Module
            if relu_ready_in:
                relu_inst = relu(inputs_relu)
                loop_out = next(relu_inst, None)
                if loop_out is not None:
                    outputs_relu = loop_out
                    relu_done_out = 0
                else:
                    relu_done_out = 1

            # linear loop initial module
            if linear_init_loop_start_ready_in:
                linear_init_loop_inst = linear_layer_init_loop(
                    m_size,
                    bias_base_addr,
                    output_base_addr)
            if linear_init_loop_next_ready_in:
                loop_out = next(linear_init_loop_inst, None)
                if loop_out is not None:
                    linear_init_loop_bias_addr, linear_init_loop_output_addr, = loop_out
                    linear_init_loop_ready_out = 1
                    linear_init_loop_done_out = 0
                else:
                    linear_init_loop_ready_out = 0
                    linear_init_loop_done_out = 1
            
            # linear loop mac module
            if linear_mac_loop_start_ready_in:
                linear_mac_loop_inst = linear_layer_mac_loop(
                    m_size,
                    chw_size,
                    input_base_addr,
                    weight_base_addr,
                    output_base_addr)
            if linear_mac_loop_next_ready_in:
                loop_out = next(linear_mac_loop_inst, None)
                if loop_out is not None:
                    linear_mac_loop_ready_out = 1
                    linear_mac_loop_done_out = 0
                    linear_mac_loop_input_addr, linear_mac_loop_weight_addr, linear_mac_loop_output_addr = loop_out                
                else:
                    linear_mac_loop_ready_out = 0
                    linear_mac_loop_done_out = 1

            # bram modules
            if bram0_read_enable:
                bram0_read_out = self.bram.read_bank(0, bram0_read_addr)
            if bram1_read_enable:
                bram1_read_out = self.bram.read_bank(1, bram1_read_addr)
            if bram1_write_enable:
                self.bram.write_bank(1, bram1_write_addr, bram1_write_val)
            
            # actual fsm
            if next_layer:
                next_layer = 0

                # emulation of generated verilog
                if layer_num == len(layers):
                    break

                layer_exec_info = layers[layer_num]
                layer_type = layer_exec_info['layer_type']
                layer_config = layer_exec_info['config']
                if layer_type == LayerType.DENSE:
                    input_base_addr = layer_config['input_base_addr']
                    weight_base_addr = layer_config['weight_base_addr']
                    bias_base_addr = layer_config['bias_base_addr']
                    output_base_addr = layer_config['output_base_addr']

                    m_size = layer_config['m_size']
                    chw_size = layer_config['chw_size']
                    linear_layer_step = LayerStep.INIT_BIAS if layer_config['has_bias'] else LayerStep.MAC_OUTPUT

                elif layer_type == LayerType.RELU:
                    output_base_addr = layer_config['output_base_addr']
                    n_size = layer_config['n_size'] 

                elif layer_type == LayerType.MOVE:
                    input_base_addr = layer_config['input_base_addr']
                    output_base_addr = layer_config['output_base_addr']

                    n_size = layer_config['n_size']

                elif layer_type == LayerType.OUTPUT:
                    output_base_addr = layer_config['output_base_addr']

                    n_size = layer_config['n_size']
                
                # next layer reset variables common between layers
                bram0_read_addr = 0
                bram0_read_enable = 0
                bram0_read_out = 0
                
                bram1_read_enable = 0
                bram1_read_addr = 0
                bram1_read_out = 0
                bram1_write_enable = 0
                bram1_write_addr = 0
                bram1_write_val = 0

                mac_ready_in = 0
                layer_num = layer_num + 1
            
            # execution of the layers
            elif layer_type == LayerType.DENSE:
                if linear_layer_step == LayerStep.INIT_BIAS:
                    if linear_init_loop_ready_out:
                        bram0_read_addr = linear_init_loop_bias_addr
                        bram0_read_enable = 1
                    
                    if linear_init_loop_ready_out or linear_init_loop_done_out:
                        bram1_write_addr = linear_init_loop_output_prev_addr
                        bram1_write_enable = linear_init_loop_started
                        bram1_write_val = bram0_read_out
                        
                        linear_init_loop_output_prev_addr = linear_init_loop_output_addr
                    
                    if linear_init_loop_start_ready_in:
                        linear_init_loop_start_ready_in = 0
                    
                    if not linear_init_loop_started:
                        linear_init_loop_start_ready_in = 1
                        linear_init_loop_next_ready_in = 1
                        linear_init_loop_started = 1
                    
                    if linear_init_loop_done_out:
                        linear_layer_step = LayerStep.MAC_OUTPUT
                        linear_init_loop_bias_addr = 0
                        linear_init_loop_output_addr = 0
                        linear_init_loop_output_prev_addr = 0
                        linear_init_loop_started = 0
                        linear_init_loop_ready_out = 0
                        linear_init_loop_done_out = 0
                        linear_init_loop_start_ready_in = 0
                        linear_init_loop_next_ready_in = 0
                
                # TODO: use mac lanes to set w_mac, i_mac, b_mac
                elif linear_layer_step == LayerStep.MAC_OUTPUT:
                    if linear_mac_loop_start_ready_in:
                        linear_mac_loop_start_ready_in = 0

                    if not linear_mac_loop_started:
                        linear_mac_loop_start_ready_in = 1
                        linear_mac_loop_next_ready_in = 1
                        linear_mac_loop_started = 1
                    
                    if linear_mac_loop_read_lane_index < MAC_LANES:
                        if linear_mac_loop_read_step == 0 and linear_mac_loop_ready_out:
                            bram0_read_addr = linear_mac_loop_weight_addr
                            bram0_read_enable = 1
                            bram1_read_addr = linear_mac_loop_input_addr
                            bram1_read_enable = 1
                            linear_mac_loop_read_step = 1
                        
                        elif linear_mac_loop_read_step == 1:
                            bram0_read_enable = 0
                            bram1_read_addr = linear_mac_loop_output_addr
                            bram1_read_enable = 1
                            weights_mac[linear_mac_loop_read_lane_index] = bram0_read_out
                            inputs_mac[linear_mac_loop_read_lane_index] = bram1_read_out
                            linear_mac_loop_read_step = 2
                        
                        elif linear_mac_loop_read_step == 2:
                            if (linear_mac_loop_write_step == 1 and (linear_mac_loop_read_lane_index < linear_mac_loop_write_lane_index)) or linear_mac_loop_write_step == 0:
                                bram0_read_enable = 0
                                bram1_read_enable = 0
                                biases_mac[linear_mac_loop_read_lane_index] = bram1_read_out
                                linear_mac_loop_output_addrs[linear_mac_loop_read_lane_index] = linear_mac_loop_output_addr

                                linear_mac_loop_read_lane_index = linear_mac_loop_read_lane_index + 1
                                linear_mac_loop_read_step = 0
                                linear_mac_loop_next_ready_in = 1
                    
                    if linear_mac_loop_read_lane_index == MAC_LANES:
                        mac_ready_in = 1
                        linear_mac_loop_write_step = 1
                        linear_mac_loop_read_step = 0
                        linear_mac_loop_read_lane_index = 0
                    
                    if linear_mac_loop_write_step == 1:
                        if linear_mac_loop_write_lane_index < MAC_LANES and mac_done_out:
                            bram1_write_addr = linear_mac_loop_output_addrs[linear_mac_loop_write_lane_index]
                            bram1_write_val = outputs_mac[linear_mac_loop_write_lane_index]
                            bram1_write_enable = 1
                            linear_mac_loop_write_lane_index = linear_mac_loop_write_lane_index + 1
                        
                        if linear_mac_loop_write_lane_index == MAC_LANES: # writing completing this cycle
                            linear_mac_loop_write_step = 0
                    
                            if linear_mac_loop_done_out:
                                next_layer = 1 # Set this to go to next layer
            
            elif layer_type == LayerType.RELU:
                if loop_started == 0:
                    loop = linear_activation_loop(
                            n_size, output_base_addr)
                    use_mac = 0
                    use_relu = 1
                loop_out = next(loop, None)
                if loop_out is not None:
                    loop_ready_out = 1
                    input_addr, weight_addr, output_addr = loop_out                
                else:
                    loop_done_out = 1
                if loop_ready_out:
                    bram0_read_addr = weight_addr
                    # TODO: are we sure this is done in 1 clock cycle?? If not fix
                    i_mac[mac_lane_idx] = bram1_read_out
                    bram1_write_addr = output_prev_addr
                    bram1_read_addr = input_addr
                    if mac_done == 1:
                        bram1_write_addr = mac_output_prev_addrs[mac_lane_idx]
                        bram1_write_val = o_mac[mac_lane_idx] 
                    if mac_lane_idx == MAC_LANES - 1:
                        bram1_write_val = o_mac
                        mac_lane_idx = 0
                    else:
                        mac_lane_idx = mac_lane_idx + 1
                    mac_output_prev_addrs[mac_lane_idx] = output_addr

                if loop_started == 0:
                    loop_started = 1

                if loop_done_out == 1:
                    loop_started = 0
                    loop_done_out = 0
                    use_relu = 0
                    next_layer = 1 # Set this to go to next layer       
        
        # EMULATION OF SENDING OUTPUT DATA FROM NEURAL NETWORK ================
        # this is not meant to be translated to verilog
        output_data = [0] * n_size
        for n in range(n_size):
            self.bram.read_bank(0, output_base_addr + n)
        # =====================================================================

        return output_data

