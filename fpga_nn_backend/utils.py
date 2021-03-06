# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:26:26 2020

@author: Shahir
"""

import os
import abc
import datetime
import json

import fpga_nn_backend

class JSONDictSerializable(metaclass=abc.ABCMeta):
    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return str(self.to_dict())
    
    @abc.abstractmethod
    def to_dict(self):
        pass
    
    @abc.abstractclassmethod
    def from_dict(cls, dct):
        pass

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data):
        return cls.from_dict(json.loads(data))

    def to_bytes(self):
        return self.to_json().encode()

    @classmethod
    def from_bytes(cls, data):
        return cls.from_json(data.decode())

def number_menu(option_list):
    print("-"*60)
    
    for n in range(len(option_list)):
        print(n, ": " , option_list[n])
    
    choice = input("Choose the number corresponding to your choice: ")
    for n in range(5):        
        try: 
            choice = int(choice)
            if choice < 0 or choice > len(option_list) - 1:
                raise ValueError    
            print("-"*60 + "\n")
            return choice, option_list[choice]
        except ValueError: 
            choice = input("Invalid input, choose again: ")
    
    raise ValueError("Not recieving a valid input")

def get_rel_pkg_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(fpga_nn_backend.__file__), "..", path))

def get_timestamp_str(include_seconds=True):
    if include_seconds:
        return datetime.datetime.now().strftime("%m-%d-%Y %I-%M-%S %p")
    else:
        return datetime.datetime.now().strftime("%m-%d-%Y %I-%M %p")