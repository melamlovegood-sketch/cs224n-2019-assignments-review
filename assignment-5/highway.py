#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """A gated highway layer that preserves the input dimensionality."""

    def __init__(self, embed_size):
        """Initialize the projection and transform-gate linear layers."""
        super(Highway, self).__init__()
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)

    def forward(self, x_conv_out):
        """Combine the nonlinear projection with the residual input."""
        x_proj = F.relu(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        return x_gate * x_proj + (1.0 - x_gate) * x_conv_out


### END YOUR CODE 
