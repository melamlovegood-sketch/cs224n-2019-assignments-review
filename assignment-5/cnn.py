#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Map character embeddings for a batch of words to word embeddings."""

    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        """Initialize the single-layer character convolution."""
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=word_embed_size,
            kernel_size=kernel_size,
        )

    def forward(self, x_reshaped):
        """Return ReLU-activated max-pooled features for each input word."""
        x_conv = F.relu(self.conv(x_reshaped))
        return torch.max(x_conv, dim=2)[0]


### END YOUR CODE
