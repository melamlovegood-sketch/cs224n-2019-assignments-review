#!/usr/bin/env python3
"""Focused tests for the Assignment 5 modules not covered deeply by sanity_check.py."""

import unittest

import torch

from cnn import CNN
from highway import Highway
from model_embeddings import ModelEmbeddings
from utils import pad_sents_char
from vocab import VocabEntry


class Assignment5ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_highway_matches_manual_formula_and_backpropagates(self):
        layer = Highway(2)
        with torch.no_grad():
            layer.projection.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.5]]))
            layer.projection.bias.copy_(torch.tensor([0.25, -0.5]))
            layer.gate.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, -0.5]]))
            layer.gate.bias.zero_()

        x = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], requires_grad=True)
        expected_projection = torch.relu(x @ layer.projection.weight.T + layer.projection.bias)
        expected_gate = torch.sigmoid(x @ layer.gate.weight.T + layer.gate.bias)
        expected = expected_gate * expected_projection + (1 - expected_gate) * x

        actual = layer(x)
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cnn_matches_manual_convolution_relu_and_max_pool(self):
        layer = CNN(char_embed_size=1, word_embed_size=1, kernel_size=5)
        with torch.no_grad():
            layer.conv.weight.fill_(1.0)
            layer.conv.bias.fill_(-3.0)

        x = torch.tensor([[[1.0, 0.0, 2.0, 1.0, 0.0, 3.0]]], requires_grad=True)
        actual = layer(x)
        # Windows sum to 4 and 6; after bias and ReLU they are 1 and 3.
        self.assertEqual(tuple(actual.shape), (1, 1))
        self.assertTrue(torch.allclose(actual, torch.tensor([[3.0]])))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_character_padding_truncation_and_no_aliasing(self):
        padded = pad_sents_char([[[1, 2], list(range(30))], [[3]]], char_pad_token=0)
        self.assertEqual((len(padded), len(padded[0]), len(padded[0][0])), (2, 2, 21))
        self.assertEqual(padded[0][0][:2], [1, 2])
        self.assertEqual(padded[0][1], list(range(21)))
        self.assertEqual(padded[1][1], [0] * 21)
        padded[1][1][0] = 9
        self.assertEqual(padded[0][0][0], 1)

    def test_model_embeddings_batch_shape_and_eval_determinism(self):
        vocab = VocabEntry()
        module = ModelEmbeddings(embed_size=8, vocab=vocab)
        inputs = vocab.to_input_tensor_char([['hello', 'x'], ['world']], torch.device('cpu'))
        module.eval()
        first = module(inputs)
        second = module(inputs)
        self.assertEqual(tuple(first.shape), (2, 2, 8))
        self.assertTrue(torch.equal(first, second))
        first.sum().backward()
        self.assertIsNotNone(module.char_embeddings.weight.grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
