import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


def confmat_plot(confmat, class_list):
    fig, ax = plt.subplots(figsize=(12,8)) 
    ax = sns.heatmap(confmat.detach().cpu().numpy(), 
                    cmap="YlGnBu", 
                    annot=True, 
                    fmt=".2f", 
                    linewidths=2, 
                    square=True,
                    xticklabels=class_list,
                    yticklabels=class_list,)

    ax.set_xlabel('Predictions', family='Arial')
    ax.set_ylabel('Groundtruths', family='Arial')
    plt.tight_layout()
    return fig


def precision_recall_plot(precision, recall, class_list):
    fig, ax = plt.subplots(figsize=(12,8)) 
    # classes = []
    # for i in range(len(precision)):
    #     classes.extend([class_list[i]] * len(precision[i]))
    precision = torch.concat(precision, 0).detach().cpu().numpy()
    recall = torch.concat(recall, 0).detach().cpu().numpy()
    data = np.array([precision, recall], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["precision", "recall"])
    sns.lineplot(data, x="recall", y="precision", ax=ax)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return fig

def precision_confidence_plot(precision, confidence, class_list):
    fig, ax = plt.subplots(figsize=(12,8)) 
    # classes = []
    # for i in range(len(precision)):
    #     classes.extend([class_list[i]] * len(precision[i]))
    precision = torch.concat(precision, 0).detach().cpu().numpy()
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([precision, confidence], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["precision", "confidence"])
    sns.lineplot(data, x="confidence", y="precision", ax=ax)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return fig



def recall_confidence_plot(recall, confidence, class_list):
    fig, ax = plt.subplots(figsize=(12,8)) 
    # classes = []
    # for i in range(len(recall)):
    #     classes.extend([class_list[i]] * len(recall[i]))
    recall = torch.concat(recall, 0).detach().cpu().numpy()
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([recall, confidence], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["recall", "confidence"])
    sns.lineplot(data, x="confidence", y="recall", ax=ax)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return fig

def specificity_confidence_plot(specificity, confidence, class_list):
    fig, ax = plt.subplots(figsize=(12,8)) 
    # classes = []
    # for i in range(len(specificity)):
    #     classes.extend([class_list[i]] * len(specificity[i]))
    specificity = torch.concat(specificity, 0).detach().cpu().numpy()
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([specificity, confidence], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["specificity", "confidence"])
    sns.lineplot(data, x="confidence", y="specificity", ax=ax)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return fig

def f1score_confidence_plot(precision, recall, confidence, class_list):
    fig, ax = plt.subplots(figsize=(12,8)) 
    # classes = []
    # for i in range(len(precision)):
    #     classes.extend([class_list[i]] * len(precision[i]))
    precision = torch.concat(precision, 0)
    recall = torch.concat(recall, 0)
    f1score = 2 * precision * recall / (precision + recall + 1e-8)
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([f1score, confidence], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["f1score", "confidence"])
    sns.lineplot(data, x="confidence", y="f1score", ax=ax)
    plt.xlim(0,1)
    plt.ylim(0,1)
    return fig

def decomp_out_plot(decomp_out):
    n, d, l = decomp_out.shape
    decomp_out = decomp_out.detach().cpu().numpy()
    fig, ax = plt.subplots(n, d,constrained_layout=True, figsize=(20,9))
    for i in range(n):
        for j in range(d):
            sns.lineplot(decomp_out[j, i, :], ax=ax[j, i])
    return fig


def diff_out_plot(diff_out):
    diff_out = diff_out[0].detach().cpu().numpy()
    d, o, l = diff_out.shape
    fig, ax = plt.subplots(o, d, constrained_layout=True, figsize=(20,9))
    for i in range(d):
        for j in range(o):
            sns.lineplot(diff_out[j, i, :], ax=ax[j, i])
    return fig


def mixer_out_plot(mixer_out):
    n, d, l = mixer_out.shape
    mixer_out = mixer_out.detach().cpu().numpy()
    mixer_out = mixer_out.reshape(n * d, l)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(7,3))
    sns.heatmap(mixer_out,  ax=ax)
    return fig


def feats_plot(feats):
    feats = feats.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(7,3))
    sns.heatmap(feats,  ax=ax)
    return fig

