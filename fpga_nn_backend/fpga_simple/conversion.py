# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:35:25 2021

@author: Shahir
"""

import struct
import math
import copy
from enum import Enum

import numpy as np

from fpga_nn_backend.fpga_simple.emulation import (
    serial_iterators,
    LayerType,
    IterationStrategy
    )

COE_INIT = """\
memory_initialization_radix=2;
memory_initialization_vector=
"""

def get_bin_string(x):
    raw = bin(x)[2:]
    return raw.zfill(8)

class ConverterLayerType(Enum):
    DENSE = 0
    CONV = 1
    RELU = 2
    RELU6 = 3
    SUM = 4
    FLATTEN = 5
    OUTPUT = 7

class ConvertedNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.parameters_info = {
            'parameters': []
        }
        self.layers_info = {
            'layers': []
        }
        # tuples containing shape
        self._current_stack = [
            input_shape
        ]
        self._iteration_strategy = IterationStrategy.SERIAL
        self._frozen = False
    
    def _add_layer(self,
                   layer_type,
                   input_shapes,
                   output_shape,
                   stack_input_indices,
                   stack_output_index,
                   parameters=None,
                   metadata=None):
        """
        Each layer is a node in the DAG with incoming edges according
        to the input indices, and the output index can be anything.
        If the output index is less than the number of elements on the stack
        prior stack elements are deleted. 
        """
        assert not self._frozen
        
        # layer type
        assert isinstance(layer_type, ConverterLayerType)

        # parameters maps parameter name to parameter index
        assert parameters is None or isinstance(parameters, dict)
        if parameters is not None:
            for k, v in parameters.items():
                assert isinstance(k, str)
                assert isinstance(v, int)
                assert 0 <= v < len(self.parameters_info['parameters'])
        
        # input and output shape for the layer
        assert isinstance(input_shapes, tuple)
        for t in input_shapes:
            assert isinstance(t, tuple)
        assert isinstance(output_shape, tuple)

        # output stack index, where layer output is dumped in the BRAM
        assert isinstance(stack_output_index, int)
        assert stack_output_index >= 0
        assert stack_output_index <= len(self._current_stack)
        
        # input stack indices, where layer input data is sourced from BRAM
        for i in stack_input_indices:
            assert isinstance(i, int)
            assert i >= 0
            assert i < len(self._current_stack)
            assert stack_output_index >= i
        
        layer_info = {
            'layer_type': layer_type,
            'input_shapes': input_shapes,
            'output_shape': output_shape,
            'output_size': np.prod(output_shape),
            'stack_input_indices': stack_input_indices,
            'stack_output_index': stack_output_index,
            'parameters': parameters,
            'metadata': metadata
        }
        
        # perform the stack add, and remove all elements after
        # which enforces it to be a DAG
        self._current_stack.insert(stack_output_index, output_shape)
        self._current_stack = self._current_stack[:stack_output_index + 1]
        
        self.layers_info['layers'].append(layer_info) 
        
        return layer_info
    
    def _add_parameter(self,
                       data):
        assert not self._frozen
        assert data.dtype == np.int8
        
        self.parameters_info['parameters'].append(data)
        weight_index = len(self.parameters_info) - 1
        
        return weight_index

    def add_flatten_layer(self,
                          input_shape,
                          stack_input_index,
                          stack_output_index):
        output_shape = (np.prod(input_shape),)

        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert stack_input_index == stack_output_index

        self._add_layer(
            ConverterLayerType.FLATTEN,
            (input_shape,),
            output_shape,
            (stack_input_index,),
            stack_output_index)

    def add_dense_layer(self,
                        input_shape,
                        output_shape,
                        stack_input_index,
                        stack_output_index,
                        weight,
                        bias=None):
        assert len(weight.shape) == 2
        if bias is not None:
            assert len(bias.shape) == 1
        
        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert weight.shape[1] == input_shape[0]
        assert output_shape[0] == weight.shape[0]
        if bias is not None:
            assert bias.shape[0] == weight.shape[0]
        
        w_index = self._add_parameter(weight)
        if bias is not None:
            b_index = self._add_parameter(bias)
        
        parameters = {
            'weight': w_index
        }
        if bias is not None:
            parameters['bias'] = b_index
        metadata = {
            'has_bias': bias is not None
        }
        
        self._add_layer(
            ConverterLayerType.DENSE,
            (input_shape,),
            output_shape,
            (stack_input_index,),
            stack_output_index,
            parameters,
            metadata)
    
    def add_conv_layer(self,
                       input_shape,
                       output_shape,
                       stack_input_index,
                       stack_output_index,
                       groups,
                       weight,
                       bias=None):
        assert len(weight.shape) == 4
        if bias is not None:
            assert len(bias.shape) == 1

        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape

        assert weight.shape[1] == (input_shape[0] // groups)
        assert output_shape[0] == weight.shape[0]
        if bias is not None:
            assert bias.shape[0] == weight.shape[0]
        
        w_index = self._add_parameter(weight)
        if bias is not None:
            b_index = self._add_parameter(bias)
        
        parameters = {
            'weight': w_index
        }
        if bias is not None:
            parameters['bias'] = b_index
        metadata = {
            'has_bias': bias is not None
        }
        
        self._add_layer(
            ConverterLayerType.CONV,
            (input_shape,),
            output_shape,
            (stack_input_index,),
            stack_output_index,
            parameters,
            metadata)
        
    def add_relu_layer(self,
                       input_shape,
                       stack_input_index,
                       stack_output_index):
        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert stack_input_index == stack_output_index
        output_shape = input_shape

        self._add_layer(
            ConverterLayerType.RELU,
            (input_shape,),
            output_shape,
            (stack_input_index,),
            stack_output_index)
    
    def add_relu6_layer(self,
                        input_shape,
                        stack_input_index,
                        stack_output_index):
        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert stack_input_index == stack_output_index
        output_shape = input_shape

        self._add_layer(
            ConverterLayerType.RELU6,
            (input_shape,),
            output_shape,
            (stack_input_index,),
            stack_output_index)
    
    def add_sum_layer(self,
                      input_shape,
                      stack_input_index1,
                      stack_input_index2,
                      stack_output_index):
        stack_input_shape1 = self._current_stack[stack_input_index1]
        assert stack_input_shape1 == input_shape
        stack_input_shape2 = self._current_stack[stack_input_index2]
        assert stack_input_shape2 == input_shape
        output_shape = input_shape

        self._add_layer(
            ConverterLayerType.SUM,
            (input_shape, input_shape),
            output_shape,
            (stack_input_index1, stack_input_index2),
            stack_output_index)
    
    def add_output_layer(self,
                        input_shape,
                        stack_input_index,
                        stack_output_index):
        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert stack_input_index == stack_output_index
        output_shape = input_shape

        self._add_layer(
            ConverterLayerType.OUTPUT,
            (input_shape,),
            output_shape,
            (stack_input_index,),
            stack_output_index)
    
    def set_iteration_strategy(self, strategy):
        assert isinstance(strategy, IterationStrategy)
        self._iteration_strategy = strategy
    
    def finalize(self):
        self._frozen = True
        assert self.layers_info['layers'][-1]['layer_type'] == ConverterLayerType.OUTPUT
    
    def get_layer_info(self):
        # information needed to write verilog for all the layers
        return copy.deepcopy(self.layers_info)
    
    def generate_parameter_data(self):
        param_datas = []
        for param in self.parameters_info['parameters']:
            param_uint8 = param.astype(np.uint8)
            param_data = bytearray()
            if self._iteration_strategy == IterationStrategy.SERIAL:
                iterator = serial_iterators[param.ndim](param.shape)
            else:
                raise NotImplementedError()
            for ind in iterator:
                v = param_uint8[ind]
                param_data.append(v)
            param_data = bytes(param_data)
            param_datas.append(param_data)
        
        return param_datas
    
    def generate_parameter_coe(self):
        param_datas = self.generate_parameter_data()
        param_datas = b''.join(param_datas)
        byte_datas = bytearray()
        num_bytes_per_line = 4
        for b in param_datas:
            byte_datas += struct.pack('>i', b)
        total_weight_data = b'\xff' + bytes(byte_datas)
        bin_str_data = bin(int.from_bytes(total_weight_data, byteorder='big'))
        bin_str_data = bin_str_data[10:]
        num_bits_per_line = num_bytes_per_line * 8
        bin_str_data = [bin_str_data[i:i+num_bits_per_line] for i in range(0, len(bin_str_data), num_bits_per_line)]
        bin_str_data = ",\n".join(bin_str_data) + ";"
        
        return COE_INIT + bin_str_data
    
    def _get_param_addr(self, param_datas, param_index):
        return sum([len(param_datas[i]) for i in range(param_index)])

    def generate_execution_info(self, scratchpad_size):
        assert self._frozen

        num_layers = len(self.layers_info['layers'])

        param_datas = self.generate_parameter_data()
        exec_layer_infos = []

        layer_stack = [
            np.prod(self.input_shape),
        ]

        max_addr_used = 0

        # start from last layer, go backwards
        for i in range(0, num_layers):
            # flatten layers do not actually compute anything
            layer_info = self.layers_info['layers'][i] 
            layer_type = layer_info['layer_type']

            last_layer = False
            move_needed = False
            move_output_index = None
            move_size = None

            if layer_type == ConverterLayerType.FLATTEN:
                n_size = layer_info['output_shape']
                input_index = layer_info['stack_input_indices'][0]
                output_index = layer_info['stack_output_index']

                assert input_index == output_index # move not supported yet with flatten

            elif layer_type == ConverterLayerType.DENSE:
                layer_exec_info = {
                    'layer_type': LayerType.DENSE,
                    'config': None
                }
                linear_config = {
                    'has_bias': None,
                    'input_base_addr': None,
                    'weight_base_addr': None,
                    'bias_base_addr': None,
                    'output_base_addr': None,
                    'm_size': None,
                    'chw_size': None,
                }
                
                linear_config['m_size'] = layer_info['output_shape'][0]
                linear_config['chw_size'] = layer_info['input_shapes'][0][0]
                
                weight_param_index = layer_info['parameters']['weight']
                linear_config['weight_base_addr'] = self._get_param_addr(param_datas, weight_param_index)

                has_bias = layer_info['metadata']['has_bias']
                if has_bias:
                    bias_param_index = layer_info['parameters']['weight']
                    linear_config['bias_base_addr'] = self._get_param_addr(param_datas, bias_param_index)
                else:
                    linear_config['bias_base_addr'] = 0
                
                input_index = layer_info['stack_input_indices'][0]
                output_index = len(layer_stack)
                linear_config['input_base_addr'] = sum([layer_stack[i] for i in range(input_index)])
                linear_config['output_base_addr']  = sum([layer_stack[i] for i in range(output_index)])
                layer_stack.append(linear_config['m_size'])

                layer_exec_info['config'] = linear_config
                exec_layer_infos.append(layer_exec_info)

                max_addr_used = max(max_addr_used, linear_config['output_base_addr'] + linear_config['m_size'])

                final_output_index = layer_info['stack_output_index']
                if final_output_index < (len(layer_stack) - 1):
                    input_addr = sum([layer_stack[i] for i in range(len(layer_stack) - 1)])
                    output_addr = sum([layer_stack[i] for i in range(final_output_index)])
                    assert output_addr < input_addr
                    layer_exec_info = {
                        'layer_type': LayerType.MOVE,
                        'config': None
                    }
                    move_config = {
                        'input_base_addr': input_addr,
                        'output_base_addr': output_addr,
                        'n_size': linear_config['m_size']
                    }

                    layer_stack.insert(final_output_index, linear_config['m_size'])
                    layer_stack = layer_stack[:final_output_index + 1]

                    layer_exec_info['config'] = move_config
                    exec_layer_infos.append(layer_exec_info)

                    max_addr_used = max(max_addr_used, move_config['output_base_addr'] + move_config['n_size'])
            
            elif layer_type == ConverterLayerType.RELU:
                layer_exec_info = {
                    'layer_type': LayerType.RELU,
                    'config': None
                }
                relu_config = {
                    'input_base_addr': None,
                    'output_base_addr': None,
                    'n_size': None,
                }
                
                act_size = np.prod(layer_info['input_shapes'][0])
                relu_config['n_size'] = act_size

                input_index = layer_info['stack_input_indices'][0]
                output_index = layer_info['stack_output_index']
                relu_config['input_base_addr'] = sum([layer_stack[i] for i in range(input_index)])
                relu_config['output_base_addr'] = sum([layer_stack[i] for i in range(output_index)])

                assert input_index == output_index

                layer_exec_info['config'] = relu_config
                exec_layer_infos.append(layer_exec_info)

                max_addr_used = max(max_addr_used, relu_config['output_base_addr'] + relu_config['n_size'])

            elif layer_type == ConverterLayerType.OUTPUT:
                layer_exec_info = {
                    'layer_type': LayerType.OUTPUT,
                    'config': None
                }
                output_config = {
                    'output_base_addr': None,
                    'n_size': None
                }

                output_config['output_base_addr'] = sum([layer_stack[i] for i in range(len(layer_stack) - 1)])
                output_config['n_size'] = np.prod(layer_info['output_shape'])                

                layer_exec_info['config'] = output_config
                exec_layer_infos.append(layer_exec_info)
                last_layer = True

                max_addr_used = max(max_addr_used, output_config['output_base_addr'] + output_config['n_size'])      
            
            scratchpad_used = sum(layer_stack)
            if scratchpad_used > scratchpad_size:
                raise ValueError("Not enough scratpad space")

            if last_layer:
                break
        
        layer_exec_info = {
            'layer_type': LayerType.INPUT_RESET,
            'config': None
        }
        reset_config = {
            'output_base_addr': None,
            'n_size': None
        }
        input_size = np.prod(self.input_shape)
        reset_config['output_base_addr'] = input_size
        reset_config['n_size'] = max_addr_used + 1 - input_size
        layer_exec_info['config'] = reset_config
        exec_layer_infos.insert(0, layer_exec_info)

        exec_info = {}
        exec_info['input_shape'] = self.input_shape
        exec_info['inital_input_addr'] = 0
        exec_info['layers'] = exec_layer_infos

        return exec_info        
    
    def generate_verilog(self):
        verilog = """
        linear_layer_mac_loop(.clk_in(clock), m_size_in({0}))
        """.format(784)
    
