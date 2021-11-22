# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:40 2021

@author: Shahir
"""

from enum import Enum

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

class ImageDatasetType(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

IMG_DATASET_TO_NUM_CLASSES = {
    ImageDatasetType.MNIST: 10,
    ImageDatasetType.CIFAR10: 10,
    ImageDatasetType.CIFAR100: 100
}

IMG_DATASET_TO_IMG_SIZE = {
    ImageDatasetType.MNIST: 28,
    ImageDatasetType.CIFAR10: 32,
    ImageDatasetType.CIFAR100: 32
}

IMG_DATASET_TO_IMG_SIZE_FLAT = {
    ImageDatasetType.MNIST: 28**2,
    ImageDatasetType.CIFAR10: 3*32**2,
    ImageDatasetType.CIFAR100: 3*32**2
}

IMG_DATASET_TO_NUM_SAMPLES = {
    ImageDatasetType.MNIST: (60000, 10000),
    ImageDatasetType.CIFAR10: (50000, 10000),
    ImageDatasetType.CIFAR100: (50000, 10000)
}

class RawDataset(Dataset):
    def __init__(self,
                 x_data,
                 y_data=None,
                 metadata=None):
        self.labeled = y_data is not None
        self.x_data = x_data
        self.y_data = y_data
        self.metadata = metadata

    def __getitem__(self, index):
        if self.labeled:
            return (self.x_data[index], self.y_data[index])
        
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)

class TransformDataset(Dataset):
    def __init__(self,
                 dataset,
                 labeled=True,
                 transform_func=None):
        self.dataset = dataset
        self.transform_func = transform_func
        self.labeled = labeled

    def __getitem__(self, index):
        if self.labeled:
            inputs, label = self.dataset[index]
            if self.transform_func:
                inputs = self.transform_func(inputs)
            return (inputs, label)
        else:
            inputs = self.dataset[index]
            inputs = self.transform_func(inputs)
            return inputs
    
    def __len__(self):
        return len(self.dataset)

def apply_img_transforms(datasets,
                         dataset_type,
                         new_input_size=None,
                         augment=False,
                         flatten=False):
    if dataset_type == ImageDatasetType.MNIST:
        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10)], p=0.9),
            transforms.ToTensor()])
    
        test_transform = transforms.Compose([
            transforms.ToTensor()])
    
    elif dataset_type == ImageDatasetType.CIFAR10 or dataset_type == ImageDatasetType.CIFAR100:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10)], p=0.9),
            transforms.ToTensor()])
        
        test_transform = transforms.Compose([
            transforms.ToTensor()])
    else:
        raise ValueError()
    
    if not augment:
        train_transform = test_transform
    
    if new_input_size:
        train_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            train_transform])
        test_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            test_transform])   

    if flatten:
        flatten = transforms.Lambda(torch.flatten)
        train_transform = transforms.Compose([
            train_transform,
            flatten])
        test_transform = transforms.Compose([
            test_transform,
            flatten])   

    train_dataset = TransformDataset(
        datasets['train'],
        transform_func=train_transform)
    test_dataset = TransformDataset(
        datasets['test'],
        transform_func=test_transform)
    
    return {'train': train_dataset,
            'test': test_dataset}

def get_img_dataset(data_dir,
                    dataset_type,
                    num_train_samples=None,
                    num_test_samples=None):
    if dataset_type == ImageDatasetType.MNIST:
        train_dataset = torchvision.datasets.MNIST(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.MNIST(
            data_dir,
            train=False,
            download=True)
    elif dataset_type == ImageDatasetType.CIFAR10:
        train_dataset = torchvision.datasets.CIFAR10(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir,
            train=False,
            download=True)
    elif dataset_type == ImageDatasetType.CIFAR100:
        train_dataset = torchvision.datasets.CIFAR100(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            data_dir,
            train=False,
            download=True)
    else:
        raise ValueError()
    
    if num_train_samples:
        assert num_train_samples <= len(train_dataset)
        train_dataset = Subset(train_dataset, list(range(num_train_samples)))
    if num_test_samples:
        assert num_test_samples <= len(test_dataset)
        test_dataset = Subset(test_dataset, list(range(num_test_samples)))

    return {'train': train_dataset,
            'test': test_dataset}

def get_dataloaders(datasets,
                    train_batch_size,
                    test_batch_size,
                    num_workers=4,
                    pin_memory=False):
    train_dataset = datasets['train']
    train_loader = DataLoader(train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
    test_shuffle_loader = DataLoader(test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    return {'train': train_loader,
            'test': test_loader,
            'test_shuffle': test_shuffle_loader}
