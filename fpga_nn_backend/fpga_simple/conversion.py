# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:35:25 2021

@author: Shahir
"""

import math
import copy

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
        assert isinstance(layer_type, LayerType)

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

        self._add_layer(
            LayerType.FLATTEN,
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
            LayerType.DENSE,
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
            LayerType.CONV,
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
        output_shape = input_shape

        self._add_layer(
            LayerType.RELU,
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
        output_shape = input_shape

        self._add_layer(
            LayerType.RELU6,
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
            LayerType.SUM,
            (input_shape, input_shape),
            output_shape,
            (stack_input_index1, stack_input_index2),
            stack_output_index)
    
    def set_iteration_strategy(self, strategy):
        assert isinstance(strategy, IterationStrategy)
        self._iteration_strategy = strategy
    
    def finalize(self):
        self._frozen = True
    
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
        total_weight_data = b'\xff' + b''.join(param_datas)
        bin_str_data = bin(int.from_bytes(total_weight_data, byteorder='big'))
        bin_str_data = bin_str_data[10:]
        bin_str_data = [bin_str_data[i:i+8] for i in range(0, len(bin_str_data), 8)]
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

        # start from last layer, go backwards
        for i in range(0, num_layers):
            layer_info = self.layers_info['layers'][i] 
            layer_type = layer_info['layer_type']

            if layer_type == LayerType.FLATTEN:
                pass

            elif layer_type == LayerType.DENSE:
                layer_exec_info = {
                    'layer_type': layer_type,
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
                
                input_index = layer_info['stack_input_indices'][0]
                output_index = len(layer_stack)
                linear_config['input_base_addr'] = sum([layer_stack[i] for i in range(input_index)])
                linear_config['output_base_addr']  = sum([layer_stack[i] for i in range(output_index)])
                layer_stack.append(linear_config['m_size'])

                layer_exec_info['config'] = linear_config
                exec_layer_infos.append(layer_exec_info)

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
                        'move_size': linear_config['m_size']
                    }

                    layer_stack.insert(final_output_index, linear_config['m_size'])
                    layer_stack = layer_stack[:final_output_index + 1]

                    layer_exec_info['config'] = move_config
                    exec_layer_infos.append(layer_exec_info)
            
            elif layer_type == LayerType.RELU:
                layer_exec_info = {
                    'layer_type': layer_type,
                    'config': None
                }
                relu_config = {
                    'input_base_addr': None,
                    'output_base_addr': None,
                    'n_size': None,
                    'inplace': None # there for completeness, fpga doesn't need this
                }
                
                act_size = np.prod(layer_info['input_shapes'][0])
                relu_config['n_size'] = act_size

                input_index = layer_info['stack_input_indices'][0]
                output_index = layer_info['stack_output_index']
                relu_config['input_base_addr'] = sum([layer_stack[i] for i in range(input_index)])
                relu_config['output_base_addr'] = sum([layer_stack[i] for i in range(output_index)])
                relu_config['inplace'] = (input_index == output_index)

                layer_exec_info['config'] = relu_config
                exec_layer_infos.append(layer_exec_info)

            elif layer_type == LayerType.OUTPUT:
                layer_exec_info = {
                    'layer_type': layer_type,
                    'config': None
                }
                output_config = {
                    'input_base_addr': None
                }

                output_config['input_base_addr'] = sum([layer_stack[i] for i in range(len(layer_stack) - 1)])

                layer_exec_info['config'] = output_config
                exec_layer_infos.append(layer_exec_info)
            
            scratchpad_used = sum(layer_stack)
            if scratchpad_used > scratchpad_size:
                raise ValueError("Not enough scratpad space")
        
        exec_info = {}
        exec_info['input_shape'] = self.input_shape
        exec_info['scratchpad_size'] = scratchpad_size
        exec_info['layers'] = exec_layer_infos

        return exec_info        
    
    def generate_verilog(self):
        verilog = """
        linear_layer_mac_loop(.clk_in(clock), m_size_in({0}))
        """.format(784)
    
