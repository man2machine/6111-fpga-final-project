# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:35:25 2021

@author: Shahir
"""

import copy
from enum import Enum

import numpy as np

from fpga_nn_backend.fpga.emulation import (
    serial_iterators,
    bfs_iterators
    )

COE_INIT = """\
memory_initialization_radix=2;
memory_initialization_vector=
"""

def get_bin_string(x):
    raw = bin(x)[2:]
    return raw.zfill(8)

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

class ConvertedNN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.weight_storage_info = {
            'parameters': []
        }
        self.execution_info = {
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
                   input_shape,
                   output_shape,
                   stack_input_indices,
                   stack_output_index,
                   parameters=None):
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
        assert isinstance(input_shape, tuple)
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
            'input_shape': input_shape,
            'output_shape': output_shape,
            'stack_input_indices': stack_input_indices,
            'stack_output_index': stack_output_index,
            'parameters': parameters
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
                        bias):
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        
        w_index = self._add_parameter(weight)
        b_index = self._add_parameter(bias)
        
        parameters = {
            "weight": w_index,
            "bias": b_index
        }

        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert weight.shape[1] == input_shape[0]
        assert output_shape[0] == weight.shape[0]
        assert bias.shape[0] == weight.shape[0]
        
        self._add_layer(
            LayerType.DENSE,
            input_shape,
            output_shape,
            (stack_input_index,),
            stack_output_index,
            parameters)
        
    
    def add_conv_layer(self,
                       input_shape,
                       output_shape,
                       stack_input_index,
                       stack_output_index,
                       groups,
                       weight,
                       bias):
        assert len(weight.shape) == 4
        assert len(bias.shape) == 1
    
    def add_relu_layer(self,
                       input_shape,
                       output_shape,
                       stack_input_index,
                       stack_output_index):
        stack_input_shape = self._current_stack[stack_input_index]
        assert stack_input_shape == input_shape
        assert output_shape == input_shape

        self._add_layer(
            LayerType.RELU,
            input_shape,
            output_shape,
            (stack_input_index,),
            stack_output_index)
    
    def add_relu6_layer(self,
                        input_shape,
                        output_shape,
                        stack_input_index,
                        stack_output_index):
        pass
    
    def add_sum_layer(self,
                      input_shape,
                      output_shape,
                      stack_input_index1,
                      stack_input_index2,
                      stack_output_index):
        pass
    
    def set_iteration_strategy(self, strategy):
        assert isinstance(strategy, IterationStrategy)
        self._iteration_strategy = strategy
    
    def finalize(self):
        self._frozen = True
    
    def generate_parameter_data(self):
        param_datas = []
        for param in self.weight_storage_info['parameters']:
            param_uint8 = param.astype(np.uint8)
            param_data = bytearray()
            if self._iteration_strategy == IterationStrategy.SERIAL:
                iterator = serial_iterators[param.ndim](param.shape)
            else:
                raise NotImplementedError()
                serial_axes = NotImplemented
                iterator = bfs_iterators[param.ndim](param.shape, serial_axes)
            for ind in iterator:
                v = param_uint8[ind]
                param_data.append(v)
            param_data = bytes(param_data)
            param_datas.append(param_data)
        
        return param_datas
    
    def generate_parameter_coe(self, num_banks):
        if num_banks != 1:
            raise NotImplementedError()
        
        # in future, utilize function below

        param_datas = self.generate_parameter_data()
        total_weight_data = b'\xff' + b''.join(param_datas)
        bin_str_data = bin(int.from_bytes(total_weight_data, byteorder='big'))
        bin_str_data = bin_str_data[10:]
        bin_str_data = [bin_str_data[i:i+8] for i in range(0, len(bin_str_data), 8)]
        bin_str_data = "\n".join(bin_str_data)
        
        return COE_INIT + bin_str_data
    
    def generate_parameter_bank_info(self, num_banks):
        raise NotImplementedError()
        
        param_datas = self.generate_parameter_data()
        # generate bytes data for each BRAM bank
        # generate start addr for each weight in each BRAM bank
        # output any other necessary information so that we can create the verilog easily

        info = {
            'num_params': None,
            'param_start_addrs': [],
            'data_per_bank': [],
            'iteration_strategy': None
        }
    
    def get_execution_info(self):
        # information needed to write verilog for all the layers
        return copy.deepcopy(self.execution_info)

    def emulate_nn(self, input_data):
        pass
    
