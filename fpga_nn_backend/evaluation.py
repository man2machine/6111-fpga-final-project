# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:32 2021

@author: Shahir
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import itertools

def get_dataloader_stats(dataloader, model, criterion, device, max_iter=None):
    with torch.set_grad_enabled(False):
        model.eval()
        
        input_datas = []
        label_datas = []
        output_datas = []
        pred_datas = []
        
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        n = 0

        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct_count = (preds == labels).sum()

            input_datas.append(inputs.detach().cpu().numpy())
            label_datas.append(labels.detach().cpu().numpy())
            output_datas.append(outputs.detach().cpu().numpy())
            pred_datas.append(preds.detach().cpu().numpy())
            
            running_loss += loss.detach().item() * inputs.size(0)
            running_correct += correct_count.detach().item()
            running_count += inputs.size(0)
            avg_loss = running_loss / running_count
            avg_acc = running_correct / running_count

            n += 1
            if max_iter and n == max_iter:
                break
            
    stats = {
        "inputs": np.concatenate(input_datas),
        "labels": np.concatenate(label_datas),
        "outputs": np.concatenate(output_datas),
        "preds": np.concatenate(pred_datas),
        "loss": avg_loss,
        "acc": avg_acc
    }
    
    return stats

def plot_embedding(x_emb, y=None, default_color=True):
    cmap = plt.get_cmap('jet')
    fig, ax = plt.subplots()
    if y is not None:
        assert len(x_emb) == len(y)
        num_labels = max(y) + 1
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if num_labels <= len(default_colors) or default_color:
            colors = default_colors
        else:
            colors = [cmap(n/num_labels) for n in range(num_labels)]
        
        markers = ['o', '+', 'x', 'v', '^', '*']
        
        prop_iter = itertools.product(markers, colors)
        
        for label, props in zip(range(num_labels), prop_iter):
            ix = np.where(y == label)
            ax.scatter(x_emb[ix, 0], x_emb[ix, 1],
                       c=[props[1] for _ in ix],
                       label=label,
                       marker=props[0],
                       s=0.8)
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.scatter(x_emb[:, 0], x_emb[:, 1], s=0.8)
    plt.show()

def show_dataset_samples_img(dataset,
                             img_transform=None,
                             labeled=True,
                             tile_dims=(6, 3),
                             figsize=(8, 4),
                             cmap=None):
    fig = plt.figure(1, figsize=figsize)
    h, w = tile_dims
    for y in range(h):
        for x in range(w):
            n = y * w + x
            ax = fig.add_subplot(w, h, n + 1)
            if labeled:
                img, label = dataset[n]
            else:
                img = dataset[n]
            if img_transform:
                img = img_transform(img)
            ax.imshow(img, cmap=cmap)
            if labeled:
                ax.set_title(str(label))
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_model_predictions_img(dataloader,
                               model,
                               device,
                               inputs_transform=None,
                               labeled=True,
                               tile_dims=(3, 3),
                               figsize=(6, 6)):
    input_datas = []
    pred_datas = []
    output_datas = []
    label_datas = []

    num_imgs = np.prod(tile_dims)
    data_count = 0
    for data in tqdm(dataloader):
        if labeled:
            inputs, labels = data
        else:
            inputs = data
        if inputs_transform:
            inputs = inputs_transform(inputs)
        inputs = inputs.to(device)
        if labeled:
            labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

        outputs = torch.softmax(outputs, dim=1)
        input_datas.append(inputs.cpu().numpy())
        pred_datas.append(preds.cpu().numpy())
        output_datas.append(outputs.cpu().numpy())
        if labeled:
            label_datas.append(labels.cpu().numpy())

        data_count += inputs.size(0)
        if data_count >= np.prod(num_imgs):
            break
    
    inputs = np.concatenate(input_datas)
    preds = np.concatenate(pred_datas)
    outputs = np.concatenate(output_datas)
    if labeled:
        labels = np.concatenate(label_datas)

    fig = plt.figure(1, figsize=figsize)
    h, w = tile_dims
    for y in range(h):
        for x in range(w):
            n = y * w + x
            ax = fig.add_subplot(w, h, n + 1)
            img = inputs[n].transpose(1, 2, 0)
            ax.imshow(img)
            pred = preds[n]
            if labeled:
                label = labels[n]
                ax.set_title("True: {}\nPred: {}".format(label, pred))
            else:
                ax.set_title("Pred: {}".format(pred))
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    data = {"inputs": inputs,
            "preds": preds,
            "outputs": outputs}
    if labeled:
        data["labels"] = labels
    
    return data
