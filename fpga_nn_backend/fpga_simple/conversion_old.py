# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:35:25 2021

@author: Shahir
"""

import math
import copy

import numpy as np

from fpga_nn_backend.fpga.emulation import (
    serial_iterators,
    bfs_iterators,
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

class ConvertedNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.weight_storage_info = {
            'parameters': []
        }
        self.execution_info = {
            'layers': [],
            'layers_dag': {}
        }
        # tuples containing shape
        self._current_stack = [
            input_shape
        ]
        self._iteration_strategy = IterationStrategy.SERIAL
        self._frozen = False
    
    def _add_layer(self,
                   layer_type,
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
        assert isinstance(layer_type, LayerType)

        # parameters maps parameter name to parameter index
        assert parameters is None or isinstance(parameters, dict)
        if parameters is not None:
            for k, v in parameters.items():
                assert isinstance(k, str)
                assert isinstance(v, int)
                assert 0 <= v < len(self.weight_storage_info['parameters'])
        
        # input and output shape for the layer
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
        
        self.execution_info['layers'].append(layer_info) 
        
        return layer_info
    
    def _add_parameter(self,
                       data):
        assert not self._frozen
        assert data.dtype == np.int8
        
        self.weight_storage_info['parameters'].append(data)
        weight_index = len(self.weight_storage_info) - 1
        
        return weight_index

    def add_flatten_layer(self,
                          input_shape,
                          stack_input_index,
                          stack_output_index):
        output_shape = (np.prod(input_shape),)

        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape

        self._add_layer(
            LayerType.FLATTEN,
            input_shape,
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
            "weight": w_index
        }
        if bias is not None:
            parameters['bias'] = b_index
        metadata = {
            'has_bias': bias is not None
        }
        
        self._add_layer(
            LayerType.DENSE,
            input_shape,
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
            "weight": w_index
        }
        if bias is not None:
            parameters['bias'] = b_index
        metadata = {
            'has_bias': bias is not None
        }
        
        self._add_layer(
            LayerType.CONV,
            input_shape,
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
        output_shape = input_shape

        self._add_layer(
            LayerType.RELU,
            input_shape,
            output_shape,
            (stack_input_index,),
            stack_output_index)
    
    def add_relu6_layer(self,
                        input_shape,
                        stack_input_index,
                        stack_output_index):
        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        output_shape = input_shape

        self._add_layer(
            LayerType.RELU6,
            input_shape,
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
            LayerType.SUM,
            output_shape,
            (stack_input_index1, stack_input_index1),
            stack_output_index)
    
    def set_iteration_strategy(self, strategy):
        assert isinstance(strategy, IterationStrategy)
        self._iteration_strategy = strategy
    
    def finalize(self):
        self._frozen = True
    
    def get_execution_info(self):
        # information needed to write verilog for all the layers
        return copy.deepcopy(self.execution_info)
    
    def generate_parameter_data(self):
        param_datas = []
        for param in self.weight_storage_info['parameters']:
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
        total_weight_data = b'\xff' + b''.join(param_datas)
        bin_str_data = bin(int.from_bytes(total_weight_data, byteorder='big'))
        bin_str_data = bin_str_data[10:]
        bin_str_data = [bin_str_data[i:i+8] for i in range(0, len(bin_str_data), 8)]
        bin_str_data = ",\n".join(bin_str_data) + ";"
        
        return COE_INIT + bin_str_data
    
    def generate_full_execution_info(self, scratchpad_size):
        assert self._frozen

        info = copy.deepcopy(self.execution_info)

        stack_start = scratchpad_size
        layer_stack = [] # tuples of (start_addr, end_addr, layer_index)
        num_layers = len(info['layers'])
        layers_visited = set()

        bfs_layers = []
        # start from last layer, go backwards
        for i in range(num_layers, -1, -1):
            layer_info = info['layers'][i]
            layer_info['write']
        
        info['input_shape'] = self.input_shape
        info['output_shape'] = info['layers'][-1]['output_shape']
    
    def generate_verilog(self):
        pass
    
