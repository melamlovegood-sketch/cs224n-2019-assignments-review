#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        self.target_vocab = target_vocab
        self.vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, self.vocab_size)
        self.decoderCharEmb = nn.Embedding(
            self.vocab_size,
            char_embedding_size,
            padding_idx=target_vocab.char2id['<pad>'],
        )
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embeddings = self.decoderCharEmb(input)
        hidden_states, dec_hidden = self.charDecoder(char_embeddings, dec_hidden)
        scores = self.char_output_projection(hidden_states)
        return scores, dec_hidden
        ### END YOUR CODE 

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input_sequence = char_sequence[:-1]
        target_sequence = char_sequence[1:]
        scores, _ = self.forward(input_sequence, dec_hidden)
        loss_function = nn.CrossEntropyLoss(
            ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum'
        )
        return loss_function(
            scores.reshape(-1, self.vocab_size), target_sequence.reshape(-1)
        )
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].size(1)
        decoded_characters = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        current_char = torch.full(
            (1, batch_size),
            self.target_vocab.start_of_word,
            dtype=torch.long,
            device=device,
        )
        dec_hidden = initialStates

        for _ in range(max_length):
            scores, dec_hidden = self.forward(current_char, dec_hidden)
            current_ids = scores.squeeze(0).argmax(dim=1)
            for batch_index, char_id_tensor in enumerate(current_ids):
                if finished[batch_index]:
                    continue
                char_id = char_id_tensor.item()
                if char_id == self.target_vocab.end_of_word:
                    finished[batch_index] = True
                elif char_id == self.target_vocab.start_of_word:
                    continue
                else:
                    decoded_characters[batch_index].append(
                        self.target_vocab.id2char[char_id]
                    )
            current_char = current_ids.unsqueeze(0)
            if bool(finished.all()):
                break

        return [''.join(characters) for characters in decoded_characters]
        ### END YOUR CODE
