import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


def confmat_plot(confmat, class_list):
    fig, ax = plt.subplots(figsize=(6,4)) 
    ax = sns.heatmap(confmat, 
                    cmap="YlGnBu", 
                    annot=True, 
                    fmt=".2f", 
                    linewidths=2, 
                    square=True,
                    xticklabels=class_list,
                    yticklabels=class_list,)

    ax.set_xlabel('Prediction$', family='Arial')
    ax.set_ylabel('Groundtruth$', family='Arial')
    plt.tight_layout()
    return fig


def precision_recall_plot(precision, recall, class_list):
    fig, ax = plt.subplots(figsize=(6,4)) 
    classes = []
    for i in range(len(precision)):
        classes.extend([class_list[i]] * len(precision[i]))
    precision = torch.concat(precision, 0).detach().cpu().numpy()
    recall = torch.concat(recall, 0).detach().cpu().numpy()
    data = np.array([precision, recall, classes], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["precision", "recall", "classes"])
    sns.lineplot(data, x="recall", y="precision", hue="classes", ax=ax)
    return fig

def precision_confidence_plot(precision, confidence, class_list):
    fig, ax = plt.subplots(figsize=(6,4)) 
    classes = []
    for i in range(len(precision)):
        classes.extend([class_list[i]] * len(precision[i]))
    precision = torch.concat(precision, 0).detach().cpu().numpy()
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([precision, confidence, classes], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["precision", "confidence", "classes"])
    sns.lineplot(data, x="confidence", y="precision", hue="classes", ax=ax)
    return fig



def recall_confidence_plot(recall, confidence, class_list):
    fig, ax = plt.subplots(figsize=(6,4)) 
    classes = []
    for i in range(len(recall)):
        classes.extend([class_list[i]] * len(recall[i]))
    recall = torch.concat(recall, 0).detach().cpu().numpy()
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([recall, confidence, classes], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["recall", "confidence", "classes"])
    sns.lineplot(data, x="confidence", y="recall", hue="classes", ax=ax)
    return fig


def f1score_confidence_plot(precision, recall, confidence, class_list):
    fig, ax = plt.subplots(figsize=(6,4)) 
    classes = []
    for i in range(len(precision)):
        classes.extend([class_list[i]] * len(precision[i]))
    precision = torch.concat(precision, 0)
    recall = torch.concat(recall, 0)
    f1score = 2 * precision * recall / (precision + recall + 1e-8)
    confidence = torch.concat(confidence, 0).detach().cpu().numpy()
    data = np.array([f1score, confidence, classes], dtype=np.object_)
    data = pd.DataFrame(data.transpose(), columns=["f1score", "confidence", "classes"])
    sns.lineplot(data, x="confidence", y="f1score", hue="classes", ax=ax)
    return fig
