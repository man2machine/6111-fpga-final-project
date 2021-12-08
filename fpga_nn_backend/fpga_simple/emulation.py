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
        assert 0 <= bank < len(self.banks)
        bounds = self.banks[bank]
        assert 0 <= addr <= (bounds[1] - bounds[0])
        addr = bounds[0] + addr

        return self.read(addr)
    
    def write_bank(self, bank, addr, val):
        assert 0 <= bank < len(self.banks)
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
    w_in = w_in[:]
    i_in = i_in[:]
    b_in = b_in[:]
    yield None # first cycle copy variables
    o_out = [0] * lanes
    for i in range(lanes):
        o_out[i] = np.int8(w_in[i] * i_in[i] + b_in[i])
    for c in range(cycle_delay - 2):
        yield None
    while True:
        yield o_out[:]

def relu(o_in, cycle_delay=1):
    if o_in < 0:
        a_out = 0
    else:
        a_out = o_in
    for c in range(cycle_delay - 1):
        yield None
    while True:
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

def linear_layer_mac_loop(
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
    addrs_i = [0]*mac_lanes
    addrs_w = [0]*mac_lanes
    addrs_o = [0]*mac_lanes

    assert mac_lanes < M
    assert M1 == math.ceil(M / mac_lanes)
    CHW_M0 = CHW * MAC_LANES
    m1L = 0
    CHW_m1L_m0 = 0
    for m1 in range(M1):
        for chw in range(CHW):
            # parallel loop for MAC - start

            # m1L = m1 * mac_lanes
            # CHW_m1L_m0 = (m1L * mac_lanes + m0) * CHW
            for m0 in range(mac_lanes):
                # input
                addrs_i[m0] = chw + input_addr

                # output
                addrs_o[m0] = m1L + m0 + output_addr

                # weight
                addrs_w[m0] = CHW_m1L_m0 + chw + weight_addr
                CHW_m1L_m0 += CHW

                # o[m] = o[m] + i[chw] * w[CHW*m + chw];
            
                yield (addrs_i[m0], addrs_w[m0], addrs_o[m0])
            
            # parallel loop for MAC - end
        m1L += mac_lanes
        CHW_m1L_m0 += mac_lanes*CHW

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

def activation_loop(
    N,
    output_addr):
    # Shapes:
    # output: (N,)
    
    for n in range(N):
        # output
        addr_o = n + output_addr
            
        yield (addr_o,)

        # o[n] = activation(o[n])

# =============================================================================
# Overall FSM
# =============================================================================

class FPGAEmulator:
    def __init__(self, converted_nn, bram_reserved_size):
        self.converted_nn = converted_nn
        self.bram = BRAMEmulator(BRAM_SIZE)
        weight_data = np.array(list(b''.join(self.converted_nn.generate_parameter_data())), dtype=np.int8)
        scratchpad_size = BRAM_SIZE - len(weight_data) - bram_reserved_size
        
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
            self.bram.write_bank(1, i + self.exec_info['inital_input_addr'], input_data[ind])
            i += 1  
        # =====================================================================

        # ADDR_BITS = 32
        # SIZE_BITS = 10
        # LANE_BITS = 3
        # DATA_BITS1 = 8 signed
        # DATA_BITS2 = 32 signed

        # overall layer state variables
        layer_num = 0 # 8 bits
        layer_type = 0 # 3 bits

        # addr bases
        input_base_addr = 0 # ADDR_BITS
        weight_base_addr = 0 # ADDR_BITS
        bias_base_addr = 0 # ADDR_BITS
        output_base_addr = 0 # ADDR_BITS

        # sizes
        m_size = 0 # SIZE_BITS
        chw_size = 0 # SIZE_BITS
        n_size = 0 # SIZE_BITS

        # linear layer overall state variables
        linear_layer_step = LayerStep.INIT_BIAS # 1 bit

        # linear init loop module signals
        linear_init_loop_bias_addr = 0 # ADDR_BITS
        linear_init_loop_output_addr = 0 # ADDR_BITS
        linear_init_loop_output_prev_addr = 0 # ADDR_BITS
        linear_init_loop_started = 0 # 1 bit
        linear_init_loop_ready_out = 0 # 1 bit
        linear_init_loop_done_out = 0 # 1 bit
        linear_init_loop_start_ready_in = 0 # 1 bit
        linear_init_loop_next_ready_in = 0 # 1 bit
        linear_init_loop_first_val_read = 0 # 1 bit
        linear_init_loop_num_writes = 0 # SIZE_BITS

        # linear mac loop module signals
        linear_mac_loop_input_addr = 0 # ADDR_BITS
        linear_mac_loop_weight_addr = 0 # ADDR_BITS
        linear_mac_loop_output_addr = 0 # ADDR_BITS
        linear_mac_loop_started = 0 # 1 bit
        linear_mac_loop_ready_out = 0 # 1 bit
        linear_mac_loop_done_out = 0 # 1 bit
        linear_mac_loop_start_ready_in = 0 # 1 bit
        linear_mac_loop_next_ready_in = 0 # 1 bit
        linear_mac_loop_read_step = 0 # 2 bits
        linear_mac_loop_read_lane_index = 0 # LANE_BITS
        linear_mac_loop_write_step = 0 # 1 bit
        linear_mac_loop_write_lane_index = 0 # LANE_BITS
        linear_mac_loop_output_addrs = [0] * MAC_LANES # MAC_LANES by ADDR_BITS

        # activation loop module signals
        activation_loop_output_addr = 0 # ADDR_BITS
        activation_loop_output_prev_addr = 0 # ADDR_BITS
        activation_loop_started = 0 # 1 bit
        activation_loop_ready_out = 0 # 1 bit
        activation_loop_done_out = 0 # 1 bit
        activation_loop_start_ready_in = 0 # 1 bit
        activation_loop_next_ready_in = 0 # 1 bit
        activation_loop_num_reads = 0 # SIZE_BITS
        activation_loop_num_writes = 0 # SIZE_BITS
        activation_loop_relu_started = 0 # 1 bit

        # move loop module signals
        move_loop_input_addr = 0 # ADDR_BITS
        move_loop_output_addr = 0 # ADDR_BITS
        move_loop_output_prev_addr = 0 # ADDR_BITS
        move_loop_started = 0 # 1 bit
        move_loop_ready_out = 0 # 1 bit
        move_loop_done_out = 0 # 1 bit
        move_loop_start_ready_in = 0 # 1 bit
        move_loop_next_ready_in = 0 # 1 bit
        move_loop_first_val_read = 0 # 1 bit
        move_loop_num_writes = 0 # SIZE_BITS

        # layer signals
        next_layer = 0 # 1 bit

        # modules
        linear_init_loop_inst = None
        linear_mac_loop_inst = None
        relu_inst = None
        move_loop_inst = None
        mac_inst = None

        # mac module signals
        mac_ready_in = 0 # 1 bit
        mac_done_out = 0 # 1 bit
        weights_mac = [0] * MAC_LANES # MAC_LANES by DATA_BITS1
        inputs_mac = [0] * MAC_LANES # MAC_LANES by DATA_BITS1
        biases_mac = [0] * MAC_LANES # MAC_LANES by DATA_BITS1
        outputs_mac = [0] * MAC_LANES # MAC_LANES by DATA_BITS1
        
        # relu module signals
        relu_ready_in = 0 # 1 bit
        relu_done_out = 0 # 1 bit
        input_relu = 0 # DATA_BITS1
        output_relu = 0 # DATA_BITS1

        # BRAM parameters
        bram0_read_addr = 0 # ADDR_BITS
        bram0_read_enable = 0 # 1 bit
        bram0_read_out = 0 # DATA_BITS1

        # BRAM scratchpad
        bram1_read_enable = 0 # 1 bit
        bram1_read_addr = 0 # ADDR_BITS
        bram1_read_out = 0 # DATA_BITS1
        bram1_write_enable = 0 # 1 bit
        bram1_write_addr = 0 # ADDR_BITS
        bram1_write_val = 0 # 1 bit

        # configuration load
        layers = self.exec_info['layers']
        
        # initial reset code
        next_layer = 1

        cycles = 0
        # fsm loop
        while True:
            # modules emulation

            # MAC Module
            if mac_ready_in:
                mac_inst = mac(weights_mac, inputs_mac, biases_mac)
                mac_done_out = 0
            if mac_inst:
                loop_out = next(mac_inst, None)
                if loop_out is not None:
                    outputs_mac = loop_out
                    mac_done_out = 1
                else:
                    mac_done_out = 0
            
            # ReLU Module
            if relu_ready_in:
                relu_inst = relu(input_relu)
                relu_done_out = 0
            if relu_inst:
                loop_out = next(relu_inst, None)
                if loop_out is not None:
                    output_relu = loop_out
                    relu_done_out = 1
                else:
                    relu_done_out = 0

            # linear loop initial module
            if linear_init_loop_start_ready_in:
                linear_init_loop_inst = linear_layer_init_loop(
                    m_size,
                    bias_base_addr,
                    output_base_addr)
                linear_init_loop_done_out = 0
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
                linear_mac_loop_done_out = 0
            if linear_mac_loop_next_ready_in:
                loop_out = next(linear_mac_loop_inst, None)
                if loop_out is not None:
                    linear_mac_loop_ready_out = 1
                    linear_mac_loop_done_out = 0
                    linear_mac_loop_input_addr, linear_mac_loop_weight_addr, linear_mac_loop_output_addr = loop_out                
                else:
                    linear_mac_loop_ready_out = 0
                    linear_mac_loop_done_out = 1
            
            # activation module
            if activation_loop_start_ready_in:
                activation_loop_inst = activation_loop(
                    n_size,
                    output_base_addr)
                activation_loop_done_out = 0
            if activation_loop_next_ready_in:
                loop_out = next(activation_loop_inst, None)
                if loop_out is not None:
                    activation_loop_ready_out = 1
                    activation_loop_done_out = 0
                    activation_loop_output_addr, = loop_out                
                else:
                    activation_loop_ready_out = 0
                    activation_loop_done_out = 1
            
            # activation module
            if move_loop_start_ready_in:
                move_loop_inst = move_loop(
                    n_size,
                    input_base_addr,
                    output_base_addr)
                move_loop_done_out = 0
            if move_loop_next_ready_in:
                loop_out = next(move_loop_inst, None)
                if loop_out is not None:
                    move_loop_ready_out = 1
                    move_loop_done_out = 0
                    move_loop_input_addr, move_loop_output_addr, = loop_out                
                else:
                    move_loop_ready_out = 0
                    move_loop_done_out = 1

            # BRAM modules
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
                print(layer_config)

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
            
            # dense linear layer
            elif layer_type == LayerType.DENSE:
                # initializing layer for bias parameters
                if linear_layer_step == LayerStep.INIT_BIAS:
                    # done writing biases, reset
                    if linear_init_loop_done_out and (linear_init_loop_num_writes == m_size):
                        linear_layer_step = LayerStep.MAC_OUTPUT
                        linear_init_loop_bias_addr = 0
                        linear_init_loop_output_addr = 0
                        linear_init_loop_output_prev_addr = 0
                        linear_init_loop_started = 0
                        linear_init_loop_ready_out = 0
                        linear_init_loop_done_out = 0
                        linear_init_loop_start_ready_in = 0
                        linear_init_loop_next_ready_in = 0
                        linear_init_loop_first_val_read = 0
                        linear_init_loop_num_writes = 0
                        next_layer = 1 # go to next layer
                    
                    # de-assert ready in signal
                    if linear_init_loop_next_ready_in:
                        linear_init_loop_next_ready_in = 0
                    
                    # write bias to output
                    if (linear_init_loop_num_writes < m_size) and linear_init_loop_first_val_read:
                        bram1_write_addr = linear_init_loop_output_prev_addr
                        bram1_write_enable = 1
                        bram1_write_val = bram0_read_out
                        linear_init_loop_num_writes = linear_init_loop_num_writes + 1
                    
                    # read bias
                    if linear_init_loop_ready_out:
                        bram0_read_addr = linear_init_loop_bias_addr
                        bram0_read_enable = 1
                        linear_init_loop_next_ready_in = 1
                        linear_init_loop_first_val_read = 1
                        linear_init_loop_output_prev_addr = linear_init_loop_output_addr
                    
                    # entering this state for the first time
                    if not linear_init_loop_started:
                        linear_init_loop_start_ready_in = 1
                        linear_init_loop_next_ready_in = 1
                        linear_init_loop_started = 1
                
                # performing matrix multiplication
                elif linear_layer_step == LayerStep.MAC_OUTPUT:
                    # de-assert ready in signal
                    if linear_mac_loop_start_ready_in:
                        linear_mac_loop_start_ready_in = 0
                    
                    # de-assert ready in signal
                    if linear_mac_loop_next_ready_in:
                        linear_mac_loop_next_ready_in = 0
                    
                    # de-assert ready in signal
                    if mac_ready_in:
                        mac_ready_in = 0
                    
                    if linear_mac_loop_write_step == 0:
                        bram1_write_enable = 0
                    # write output from MAC
                    elif linear_mac_loop_write_step == 1:
                        # not done writing output from MAC and MAC has finished computation
                        if linear_mac_loop_write_lane_index < MAC_LANES and mac_done_out:
                            bram1_write_addr = linear_mac_loop_output_addrs[linear_mac_loop_write_lane_index]
                            bram1_write_val = outputs_mac[linear_mac_loop_write_lane_index]
                            bram1_write_enable = 1
                            linear_mac_loop_write_lane_index = linear_mac_loop_write_lane_index + 1
                        
                        # done writing output for MAC
                        elif linear_mac_loop_write_lane_index == MAC_LANES:
                            linear_mac_loop_write_step = 0 # reset to not writing state
                            linear_mac_loop_write_lane_index = 0
                            bram1_write_enable = 0

                            # done writing output and the loop is complete
                            if linear_mac_loop_done_out:
                                next_layer = 1 # go to next layer
                    
                    # not finished reading all of the data for each of the MAC lates
                    if (linear_mac_loop_read_lane_index < MAC_LANES) and (linear_mac_loop_write_step == 0):
                        # reading weight and input from BRAM
                        if linear_mac_loop_read_step == 0 and linear_mac_loop_ready_out:
                            bram0_read_addr = linear_mac_loop_weight_addr
                            bram0_read_enable = 1
                            bram1_read_addr = linear_mac_loop_input_addr
                            bram1_read_enable = 1
                            linear_mac_loop_read_step = 1
                        
                        # reading output from BRAM
                        elif linear_mac_loop_read_step == 1:
                            bram0_read_enable = 0
                            bram1_read_addr = linear_mac_loop_output_addr # the bias is the output in this case
                            bram1_read_enable = 1
                            weights_mac[linear_mac_loop_read_lane_index] = bram0_read_out
                            inputs_mac[linear_mac_loop_read_lane_index] = bram1_read_out
                            linear_mac_loop_read_step = 2
                        
                        # starting MAC, and initializing next read
                        elif linear_mac_loop_read_step == 2:
                            # if we are writing, we move on to reading the next lane only if reading is behind writing
                            # otherwise if writing is complete (or not started for the first time), then we can read next lane no problems
                            # this is so that we do not overwrite linear_mac_loop_output_addrs while it is being used for writing
                            # if (linear_mac_loop_write_step == 1 and (linear_mac_loop_read_lane_index < linear_mac_loop_write_lane_index)) or \
                            #     linear_mac_loop_write_step == 0:
                            if linear_mac_loop_write_step == 0:
                                bram0_read_enable = 0
                                bram1_read_enable = 0
                                biases_mac[linear_mac_loop_read_lane_index] = bram1_read_out
                                linear_mac_loop_output_addrs[linear_mac_loop_read_lane_index] = linear_mac_loop_output_addr
                                # increment to next lane, and generate new addrs from loop
                                linear_mac_loop_read_lane_index = linear_mac_loop_read_lane_index + 1
                                linear_mac_loop_read_step = 0
                                linear_mac_loop_next_ready_in = 1
                    
                    # done reading for all of the lanes
                    elif linear_mac_loop_read_lane_index == MAC_LANES:
                        mac_ready_in = 1 # execute mac
                        linear_mac_loop_write_step = 1 # start writing
                        # restart reading from first lane
                        linear_mac_loop_read_step = 0
                        linear_mac_loop_read_lane_index = 0      
                    
                    # entering this state for the first time
                    if not linear_mac_loop_started:
                        linear_mac_loop_start_ready_in = 1
                        linear_mac_loop_next_ready_in = 1
                        linear_mac_loop_started = 1
            
            # activation layer
            elif layer_type == LayerType.RELU:
                # done writing activations, reset
                if activation_loop_done_out and (activation_loop_num_writes == n_size):
                    activation_loop_output_addr = 0
                    activation_loop_output_prev_addr = 0
                    activation_loop_started = 0
                    activation_loop_ready_out = 0
                    activation_loop_done_out = 0
                    activation_loop_start_ready_in = 0
                    activation_loop_next_ready_in = 0
                    activation_loop_num_reads = 0
                    activation_loop_num_writes = 0
                    activation_loop_relu_started = 0
                    next_layer = 1 # go to next layer
                
                # de-assert ready in signal
                if activation_loop_start_ready_in:
                    activation_loop_start_ready_in = 0
                
                # de-assert ready in signal
                if activation_loop_next_ready_in:
                    activation_loop_next_ready_in = 0
                
                # de-assert ready in signal
                if relu_ready_in:
                    relu_ready_in = 0
                
                # write output of activation from relu module, if we have completed the read for the relu inpu and the relu is done
                if (activation_loop_num_writes < n_size) and (activation_loop_num_reads == (activation_loop_num_writes + 1)) and relu_done_out:
                    bram1_write_addr = activation_loop_output_prev_addr
                    bram1_write_enable = 1
                    bram1_write_val = output_relu
                    activation_loop_relu_started = 0
                    activation_loop_num_writes = activation_loop_num_writes + 1
                
                # start the relu module, if we have completed the read for the relu input and the relu module is not started
                if (activation_loop_num_writes < n_size) and (activation_loop_num_reads == (activation_loop_num_writes + 1)) and not activation_loop_relu_started:
                    bram1_read_enable = 0
                    input_relu = bram1_read_out
                    relu_ready_in = 1
                    activation_loop_relu_started = 1
                
                # read for the relu module if we have not completed all the reads, and we have an address to read,
                # and the number of reads is equal to the number of writes (so reads only goes ahead of writes by one iteration)
                # or if the reads are 1 iteration ahead of writes, we start reading only if the relu happens to complete executing this cycle
                if (activation_loop_num_reads < n_size) and activation_loop_ready_out and \
                    ((activation_loop_num_reads == activation_loop_num_writes) or \
                        (activation_loop_num_reads == (activation_loop_num_writes + 1)) and relu_done_out):
                    bram1_read_addr = activation_loop_output_addr
                    bram1_read_enable = 1
                    activation_loop_next_ready_in = 1
                    activation_loop_output_prev_addr = activation_loop_output_addr
                    activation_loop_num_reads = activation_loop_num_reads + 1
                
                # entering this state for the first time
                if not activation_loop_started:
                    activation_loop_start_ready_in = 1
                    activation_loop_next_ready_in = 1
                    activation_loop_started = 1
            
            # move data layer
            elif layer_type == LayerType.MOVE:
                # done writing moves, reset
                if move_loop_done_out and (move_loop_num_writes == m_size):
                    move_loop_output_addr = 0
                    move_loop_output_prev_addr = 0
                    move_loop_started = 0
                    move_loop_ready_out = 0
                    move_loop_done_out = 0
                    move_loop_start_ready_in = 0
                    move_loop_next_ready_in = 0
                    move_loop_first_val_read = 0
                    move_loop_num_writes = 0
                    next_layer = 1 # go to next layer
                
                # de-assert ready in signal
                if move_loop_start_ready_in:
                    move_loop_start_ready_in = 0
                
                # de-assert ready in signal
                if move_loop_next_ready_in:
                    move_loop_next_ready_in = 0
                
                # write moved value to output
                if (move_loop_num_writes < m_size) and move_loop_first_val_read:
                    bram1_write_addr = move_loop_output_prev_addr
                    bram1_write_enable = 1
                    bram1_write_val = bram1_read_out
                    move_loop_num_writes = move_loop_num_writes + 1

                # read value to move
                if move_loop_ready_out:
                    bram1_read_addr = move_loop_input_addr
                    bram1_read_enable = 1
                    move_loop_next_ready_in = 1 
                    move_loop_output_prev_addr = move_loop_output_addr
                    move_loop_first_val_read = 1
                
                # entering this state for the first time
                if not move_loop_started:
                    move_loop_start_ready_in = 1
                    move_loop_next_ready_in = 1
                    move_loop_started = 1
            
            # final output layer
            elif layer_type == LayerType.OUTPUT:
                # do nothing, this layer involves no computation
                # # it just indicates to the FSM that this is the final layer, and provides the output addr and size for the result
                break

            # if layer_type == LayerType.MOVE:
            #     print(layer_type, linear_layer_step)
            #     # print("linear_mac_loop_input_addr:",linear_mac_loop_input_addr)
            #     # print("linear_mac_loop_weight_addr:",linear_mac_loop_weight_addr)
            #     # print("linear_mac_loop_output_addr:",linear_mac_loop_output_addr)
            #     # print("linear_mac_loop_started:",linear_mac_loop_started)
            #     # print("linear_mac_loop_ready_out:",linear_mac_loop_ready_out)
            #     # print("linear_mac_loop_done_out:",linear_mac_loop_done_out)
            #     # print("linear_mac_loop_start_ready_in:",linear_mac_loop_start_ready_in)
            #     # print("linear_mac_loop_next_ready_in:",linear_mac_loop_next_ready_in)
            #     # print("linear_mac_loop_read_step:",linear_mac_loop_read_step)
            #     # print("linear_mac_loop_read_lane_index:",linear_mac_loop_read_lane_index)
            #     # print("linear_mac_loop_write_step:",linear_mac_loop_write_step)
            #     # print("linear_mac_loop_write_lane_index:",linear_mac_loop_write_lane_index)
            #     # print("linear_mac_loop_output_addrs:",linear_mac_loop_output_addrs)
            #     # print("---")
            #     print("bram0_read_addr:",bram0_read_addr)
            #     print("bram0_read_enable:",bram0_read_enable)
            #     print("bram0_read_out:",bram0_read_out)
            #     print("bram1_read_enable:",bram1_read_enable)
            #     print("bram1_read_addr:",bram1_read_addr)
            #     print("bram1_read_out:",bram1_read_out)
            #     print("bram1_write_enable:",bram1_write_enable)
            #     print("bram1_write_addr:",bram1_write_addr)
            #     print("bram1_write_val:",bram1_write_val)
            #     # print("---")
            #     # print("mac_ready_in:",mac_ready_in)
            #     # print("mac_done_out:",mac_done_out)
            #     # print("weights_mac:",weights_mac)
            #     # print("inputs_mac:",inputs_mac)
            #     # print("biases_mac:",biases_mac)
            #     # print("outputs_mac:",outputs_mac)
            #     print("---")
            #     print("move_loop_input_addr:",move_loop_input_addr)
            #     print("move_loop_output_addr:",move_loop_output_addr)
            #     print("move_loop_output_prev_addr:",move_loop_output_prev_addr)
            #     print("move_loop_started:",move_loop_started)
            #     print("move_loop_ready_out:",move_loop_ready_out)
            #     print("move_loop_done_out:",move_loop_done_out)
            #     print("move_loop_start_ready_in:",move_loop_start_ready_in)
            #     print("move_loop_next_ready_in:",move_loop_next_ready_in)
            #     print("move_loop_first_val_read:",move_loop_first_val_read)
            #     print("move_loop_num_writes:",move_loop_num_writes)
            # #     print("---")

            # if layer_num == 2:
            #     n_size = 10
            #     output_base_addr = 784
            #     output_data = [0] * n_size
            #     for n in range(n_size):
            #         output_data[n] = self.bram.read_bank(1, output_base_addr + n)
            #     # =====================================================================
            #     output_data = np.array(output_data, np.int8) 
            #     print(output_data)
            #     n_size = 10
            #     output_base_addr = 0
            #     output_data = [0] * n_size
            #     for n in range(n_size):
            #         output_data[n] = self.bram.read_bank(1, output_base_addr + n)
            #     # =====================================================================
            #     output_data = np.array(output_data, np.int8) 
            #     print(output_data)
            #     print("####")

            cycles += 1

        # EMULATION OF SENDING OUTPUT DATA FROM NEURAL NETWORK ================
        # this is not meant to be translated to verilog
        output_data = [0] * n_size
        for n in range(n_size):
            output_data[n] = self.bram.read_bank(1, output_base_addr + n)
        # =====================================================================
        output_data = np.array(output_data, np.int8)

        return output_data
