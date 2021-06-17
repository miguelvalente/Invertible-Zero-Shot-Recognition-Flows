import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.linalg import norm
import wandb
import torch.nn.functional as F
from numpy.core.numeric import Inf
import torch
import numpy as np
import time
from laspy.file import File
import pandas as pd
import plotly.graph_objects as go
from matplotlib import widgets
from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import os
import math
import laspy
import torch
import torch.nn.functional as F
from yaml import load as load_yaml

eps = 1e-8


def make_toy_graph(x, epoch, text='Toy Data', fit=False, show=False, save=False):
    """
    Used to make graphs for groundtruth and generated
    """
    plt.title(rf'Toy Data {epoch}')
    plt.xlabel(r'')
    plt.ylabel(r'')
    a = plt.scatter(x[0][:, 0], x[0][:, 1], alpha=0.5)
    b = plt.scatter(x[1][:, 0], x[1][:, 1], alpha=0.5)
    c = plt.scatter(x[2][:, 0], x[2][:, 1], alpha=0.5)
    u = plt.scatter(x[3][:, 0], x[3][:, 1], alpha=0.5)
    if fit:
        plt.axis((-2, 2, -2, 2))
    plt.legend((a, b, c, u),
               ('Seen A', 'Seen B', 'Seen C', 'Unseen'),
               scatterpoints=1,
               fontsize=12,
               bbox_to_anchor=(1.04, 1),
               borderaxespad=0,
               fancybox=True,
               shadow=True)
    if show:
        plt.show()
    if save:
        wandb.log({f'{text}.png': plt})
    plt.clf()

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_except_batch(x, num_dims=1):
    '''
    Averages all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)

def reduce_mean_masked(x, mask, axis):
    x = x * mask.float()
    m = x.sum(axis=axis) / mask.sum(axis=axis).float()
    return m


def reduce_sum_masked(x, mask, axis):
    x = x * mask.float()
    m = x.sum(axis=axis)
    return m

if __name__ == '__main__':
    x = torch.rand((2, 10, 10))
    mask = torch.ones(2, 10, 10)
    reduce_sum_masked(x, mask, axis=1)
