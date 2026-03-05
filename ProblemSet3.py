#%% ----- Setup ----- 
# Imports
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

#%% ----- Question 1 -----

#%% ----- Question 2 -----

#%% ----- Question 3 -----
softmax = nn.Softmax(dim=1)

# 4 tokens, each with dimension 1x3
X = torch.tensor([
    [1.0,  0.0, -0.5],
    [0.0, -1.0,  1.0],
    [0.5,  0.5,  0.0],
    [-0.5, -1.0,  1.0]
])

# Weight matrices for queries, keys, and values - 2 hidden dimensions
W_Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5]
])
W_K = torch.tensor([
    [-1.0,  0.0],
    [ 1.0,  0.5],
    [ 0.5, -0.5]
])
W_V = torch.tensor([
    [ 0.0, -1.0],
    [ 1.0,  0.0],
    [-0.5,  0.5]
])

# Matrix operations for efficient calculation of queries, keys, and values
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Entry (j,k) in the attention map represents how much token j attends to token k
scores = Q @ K.T
scores = torch.tensor([
    [-1.0, -0.13, -0.06, 0.25],
    [-0.75, 0.25, -0.13, 0.5],
    [-0.5, -0.75, 0.13, -0.5],
    [-0.13, 0.5, -0.13, 0.5]
])

attention_map = softmax(scores)
attention_map = torch.round(attention_map, decimals=2)

# Compute output
output = attention_map @ V

# %%
