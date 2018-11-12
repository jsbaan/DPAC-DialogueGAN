import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from EncoderDecoderAttn import Encoder, Decoder
import random

class Generator(nn.Module):

    def __init__(self, vocab_size, hidden_size, embed_size, max_len, enc_n_layers=2, \
        enc_dropout=0.2, dec_n_layers=2, dec_dropout=0.2, device='cpu'):
        super(Generator, self).__init__()

        encoder = Encoder(vocab_size, embed_size, hidden_size, enc_n_layers, enc_dropout)
        decoder = Decoder(embed_size, hidden_size, vocab_size, dec_n_layers, dec_dropout)

        self.device = device
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = autograd.Variable(torch.zeros(max_len, batch_size, vocab_size)).to(self.device)

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = autograd.Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = autograd.Variable(trg.data[t] if is_teacher else top1).to(self.device)
        return outputs

    def sample(self, context, max_len):
        """
        Samples the network using a batch of source input sequence. Passes these inputs
        through the decoder and instead of taking the top1 (like in forward), sample
        using the distribution over the vocabulary

        Inputs: dialogue context (and maximum sample sequence length
        Outputs: samples
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """
        # Initialize sample
        batch_size = context.size(1)
        vocab_size = self.decoder.output_size
        samples = autograd.Variable(torch.zeros(batch_size,max_len)).to(self.device)


        # Run input through encoder
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = autograd.Variable(context.data[0, :])  # sos

        # Pass through decoder and sample from resulting vocab distribution
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)

            # Sample token for entire batch from predicted vocab distribution
            batch_token_sample = torch.multinomial(torch.exp(output), 1).view(-1).data
            samples[:, t] = batch_token_sample
            output = autograd.Variable(batch_token_sample)
        return samples

    def sample_old(self, num_samples, max_seq_len, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.to(self.device)
            inp = inp.to(self.device)

        for i in range(max_seq_len):
            out, h = self.forward(inp, h)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size
