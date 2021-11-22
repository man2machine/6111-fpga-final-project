# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:21 2021

@author: Shahir
"""

import os
import time
import datetime
import pickle
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

from tqdm import tqdm

from fpga_nn_backend.utils import get_timestamp_str

class OptimizerType(Enum):
    SGD = "sgd"
    SGD_MOMENTUM = "sgd_momentum"
    ADAM = "adam"

class LossType(Enum):
    CROSS_ENTROPY = "cross_entropy"

class ModelTracker:
    def __init__(self, root_dir): 
        experiment_dir = "Experiment {}".format(get_timestamp_str())
        self.save_dir = os.path.join(root_dir, experiment_dir)
        self.best_model_metric = float('-inf')
        self.record_per_epoch = {}
    
    def update_info_history(self,
                            epoch,
                            info):
        os.makedirs(self.save_dir, exist_ok=True)
        self.record_per_epoch[epoch] = info
        fname = "Experiment Epoch Info History.pckl"
        with open(os.path.join(self.save_dir, fname), 'wb') as f:
            pickle.dump(self.record_per_epoch, f)
    
    def update_model_weights(self,
                             epoch,
                             model_state_dict,
                             metric=None,
                             save_best=True,
                             save_latest=True,
                             save_current=False):
        os.makedirs(self.save_dir, exist_ok=True)
        update_best = metric is None or metric > self.best_model_metric
        if update_best and metric is not None:
            self.best_model_metric = metric
        
        if save_best and update_best:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                "Weights Best.pckl"))
        if save_latest:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                "Weights Latest.pckl"))
        if save_current:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                "Weights Epoch {} {}.pckl".format(epoch, get_timestamp_str())))

def make_optimizer(model, lr=0.001, weight_decay=0.0,
                   clip_grad_norm=False, verbose=False,
                   optimzer_type=OptimizerType.SGD):
    # Get all the parameters
    params_to_update = model.parameters()

    if verbose:
        print("Params to learn:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    if optimzer_type == OptimizerType.ADAM:
        optimizer = optim.Adam(params_to_update, lr=lr,
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=True)
    elif optimzer_type == OptimizerType.SGD:
        optimizer = optim.SGD(params_to_update, lr=lr, weight_decay=weight_decay)
    elif optimzer_type == OptimizerType.SGD_MOMENTUM:
        optimizer = optim.SGD(params_to_update, lr=lr, weight_decay=weight_decay,
                              momentum=0.9)
    else:
        raise ValueError()
    if clip_grad_norm:
        nn.utils.clip_grad_norm_(params_to_update, 3.0)

    return optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer,
                     lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_scheduler(optimizer, epoch_steps, gamma):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epoch_steps, gamma=gamma)
    return scheduler

def get_loss(loss_type=LossType.CROSS_ENTROPY):
    if loss_type == LossType.CROSS_ENTROPY:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError()
    return criterion

def train_model(
        device,
        model,
        dataloaders,
        criterion,
        optimizer,
        save_dir,
        lr_scheduler=None,
        save_model=False,
        save_best=False,
        save_latest=False,
        save_all=False,
        save_log=False,
        num_epochs=1,
        early_stop=False,
        early_stop_acc=0.95,
        early_stop_patience=5):

    start_time = time.time()

    tracker = ModelTracker(save_dir)
    best_test_acc = -1
    stagnation = 0

    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        train_loss_info = {}

        print("Training")
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        train_loss_record = []
        pbar = tqdm(dataloaders['train'])
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)
                train_loss_record.append(loss.detach().item())
                loss.backward()
                optimizer.step()

                correct = torch.sum(preds == labels).item()

            running_loss += loss.detach().item() * inputs.size(0)
            running_correct += correct
            running_count += inputs.size(0)
            training_loss = running_loss / running_count
            training_acc = running_correct / running_count
            
            loss_fmt = "{:.4f}"
            desc = "Avg. Loss: {}, Total Loss: {}"
            desc = desc.format(loss_fmt.format(training_loss),
                               loss_fmt.format(loss.detach().item()))
            pbar.set_description(desc)

            del loss
        
        pbar.close()

        print("Training Loss: {:.4f}".format(training_loss))
        print("Training Accuracy: {:.4f}".format(training_acc))
        train_loss_info['loss'] = train_loss_record
        
        print("Testing")
        model.eval()
        pbar = tqdm(dataloaders['test'])
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for inputs, labels in pbar:
            running_count += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                correct = torch.sum(preds == labels).item()

            running_loss += criterion(outputs, labels).item() * inputs.size(0)
            running_correct += correct
            test_accuracy = running_correct / running_count
            test_loss = running_loss / running_count

        print("Testing loss: {:.4f}".format(test_loss))
        print("Testing accuracy: {:.4f}".format(test_accuracy))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if save_model:
            model_weights = model.state_dict()
            tracker.update_model_weights(epoch,
                                         model_weights,
                                         metric=test_accuracy,
                                         save_best=save_best,
                                         save_latest=save_latest,
                                         save_current=save_all)

        if save_log:
            info = {'train_loss_history': train_loss_info}
            tracker.update_info_history(epoch, info)

        if lr_scheduler:
            lr_scheduler.step()
        
        if early_stop:  
            print("Best testing accuracy was: {:.4f}".format(best_test_acc))
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                stagnation = 0
            elif best_test_acc > early_stop_acc:
                stagnation += 1

            if stagnation >= early_stop_patience:
                print("Ran out of patience at epoch:", epoch)
                print("Patience was:", early_stop_patience)
                break
            print("Stagnation was:", stagnation)
        
        print()

    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    return tracker

def load_weights(model,
                 weights_fname,
                 map_location=None):
    model.load_state_dict(torch.load(weights_fname, map_location=map_location))
    return model

def save_training_session(model,
                          optimizer,
                          sessions_save_dir):
    sub_dir = "Session {}".format(get_timestamp_str())
    sessions_save_dir = os.path.join(sessions_save_dir, sub_dir)
    os.makedirs(sessions_save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(sessions_save_dir, "Model State.pckl"))
    torch.save(optimizer.state_dict(), os.path.join(sessions_save_dir, "Optimizer State.pckl"))

    print("Saved session to", sessions_save_dir)

def load_training_session(model,
                          optimizer,
                          session_dir,
                          update_models=True,
                          map_location=None):
    if update_models:
        model.load_state_dict(torch.load(os.path.join(session_dir, "Model State.pckl"), map_location=map_location))
        optimizer.load_state_dict(
            torch.load(os.path.join(session_dir, "Optimizer State.pckl"), map_location=map_location))

    print("Loaded session from", session_dir)

    out_data = {'model': model,
                'optimizer': optimizer}

    return out_data

