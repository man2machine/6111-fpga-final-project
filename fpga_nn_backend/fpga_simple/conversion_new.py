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
        self.input_shape = tuple(input_shape)
        self.weight_storage_info = {
            'parameters': []
        }
        self.execution_info = {
            'layers': [],
            'layers_dag_prev': {},
            'layers_dag_next': {}
        }
        # tuples containing shape
        self._current_stack = [
            input_shape
        ]
        self._iteration_strategy = IterationStrategy.SERIAL
        self._frozen = False

        self._add_layer(
            LayerType.INPUT,
            self.input_shape,
            (0,))
    
    def _add_layer(self,
                   layer_type,
                   output_shape,
                   prev_layer_indices,
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
        
        # input layer indices, where layer input data is sourced from BRAM
        for i in prev_layer_indices:
            assert isinstance(i, int)
            assert i >= 0
            assert i < len(self.execution_info['layers'])
            # as long as each layer only depends on previous layers, its a DAG
        
        layer_info = {
            'layer_type': layer_type,
            'output_shape': output_shape,
            'output_size': np.prod(output_shape),
            'prev_layer_indices': prev_layer_indices,
            'parameters': parameters,
            'metadata': metadata
        }
        
        layer_index = len(self.execution_info['layers'])

        self.get_execution_info['layers_dag_prev'][layer_index] = list(prev_layer_indices)
        for prev_index in prev_layer_indices:
            self.get_execution_info['layers_dag_next'].setdefault(prev_index, [])
            self.get_execution_info['layers_dag_next'][prev_index].append(layer_index)
        self.execution_info['layers'].append(layer_info) 
        
        return layer_index
    
    def _add_parameter(self,
                       data):
        assert not self._frozen
        assert data.dtype == np.int8
        
        self.weight_storage_info['parameters'].append(data)
        weight_index = len(self.weight_storage_info) - 1
        
        return weight_index

    def add_flatten_layer(self,
                          input_shape,
                          prev_layer):
        output_shape = (np.prod(input_shape),)

        prev_input_shape = self.execution_info['layers'][prev_layer]['output_shape']
        assert prev_input_shape == input_shape

        return self._add_layer(
            LayerType.FLATTEN,
            output_shape,
            (prev_layer,))

    def add_dense_layer(self,
                        input_shape,
                        output_shape,
                        prev_layer,
                        weight,
                        bias=None):
        assert len(weight.shape) == 2
        if bias is not None:
            assert len(bias.shape) == 1
        
        prev_input_shape = self.execution_info['layers'][prev_layer]['output_shape']
        assert prev_input_shape == input_shape

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
            output_shape,
            (prev_layer,),
            parameters,
            metadata)
    
    def add_conv_layer(self,
                       input_shape,
                       output_shape,
                       prev_layer,
                       groups,
                       weight,
                       bias=None):
        assert len(weight.shape) == 4
        if bias is not None:
            assert len(bias.shape) == 1
        
        prev_input_shape = self.execution_info['layers'][prev_layer]['output_shape']
        assert prev_input_shape == input_shape

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
            'groups': groups,
            'has_bias': bias is not None
        }
        
        self._add_layer(
            LayerType.CONV,
            output_shape,
            (prev_layer,),
            parameters,
            metadata)
        
    def add_relu_layer(self,
                       input_shape,
                       prev_layer):
        prev_input_shape = self.execution_info['layers'][prev_layer]['output_shape']
        assert prev_input_shape == input_shape
        output_shape = input_shape

        self._add_layer(
            LayerType.RELU,
            output_shape,
            (prev_layer,))
    
    def add_relu6_layer(self,
                        input_shape,
                        prev_layer):
        prev_input_shape = self.execution_info['layers'][prev_layer]['output_shape']
        assert prev_input_shape == input_shape
        output_shape = input_shape

        self._add_layer(
            LayerType.RELU6,
            output_shape,
            (prev_layer,))
    
    def add_sum_layer(self,
                      input_shape,
                      prev_layer1,
                      prev_layer2):
        prev_input_shape1 = self.execution_info['layers'][prev_layer1]['output_shape']
        assert prev_input_shape1 == input_shape
        prev_input_shape2 = self.execution_info['layers'][prev_layer2]['output_shape']
        assert prev_input_shape2 == input_shape
        output_shape = input_shape

        self._add_layer(
            LayerType.SUM,
            output_shape,
            (prev_layer1, prev_layer2))
    
    def set_iteration_strategy(self, strategy):
        assert isinstance(strategy, IterationStrategy)
        self._iteration_strategy = strategy
    
    def finalize(self):
        self._frozen = True

    def generate_parameter_data(self):
        assert self._frozen

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
        assert self._frozen

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
            input_indices = info['layers_dag'][i]
            input_sizes = [np.prod(info['layers'][j]['output_shape']) for j in input_indices]
            output_size = np.prod(info['layers'][i]['output_shape'])
            if stack_start > 0:
                pass
        
        info['input_shape'] = self.input_shape
        info['output_shape'] = info['layers'][-1]['output_shape']

    def generate_verilog(self):
        pass
    
