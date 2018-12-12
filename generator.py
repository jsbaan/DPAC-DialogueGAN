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
import sys

class Generator(nn.Module):

    def __init__(self, vocab_size, hidden_size, embed_size, max_len, enc_n_layers=2, \
        enc_dropout=0.2, dec_n_layers=2, dec_dropout=0.2, teacher_forcing_ratio=0.5):
        super(Generator, self).__init__()


        encoder = Encoder(vocab_size, embed_size, hidden_size, enc_n_layers, enc_dropout)
        decoder = Decoder(embed_size, hidden_size, vocab_size, dec_n_layers, dec_dropout)

        self.max_len = max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = Encoder(vocab_size, embed_size, hidden_size, enc_n_layers, enc_dropout)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, dec_n_layers, dec_dropout)

    def forward(self, src, tgt):
        batch_size = src.size(1)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(self.max_len, batch_size, vocab_size)

        # TODO: Check what happens here. Hidden representation dimension is strange
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]

        sos = tgt.data[0].long()
        outputs[0,:,sos] = 1

        previous_output = sos
        for t in range(1, self.max_len):
            output, hidden, attn_weights = self.decoder(previous_output, hidden, encoder_output)
            outputs[t] = output

            is_teacher = random.random() < self.teacher_forcing_ratio
            if is_teacher:
                previous_output = tgt.data[t]
            else:
                previous_output = output.data.argmax(1)

        return outputs

    def sample(self, context):
        """
        Samples the network using a batch of source input sequence. Passes these inputs
        through the decoder and instead of taking the top1 (like in forward), sample
        using the distribution over the vocabulary


        Inputs: dialogue context and maximum sample sequence length
        Outputs: samples
        samples: num_samples x max_seq_length (a sampled sequence in each row)

        Inputs: dialogue context (and maximum sample sequence length
        Outputs: samples
            - samples: num_samples x max_seq_length (a sampled sequence in each row)"""

        # Initialize sample
        hiddens = []
        batch_size = context.size(1)
        vocab_size = self.decoder.output_size
        samples = autograd.Variable(torch.zeros(batch_size, self.max_len))
        samples_prob = autograd.Variable(torch.zeros(batch_size, self.max_len))

        # Run input through encoder
        encoder_output, hidden = self.encoder(context)
        hidden = hidden[:self.decoder.n_layers]
        hiddens.append(hidden)
        output = autograd.Variable(context.data[0, :])  # sos
        samples[:,0] = output
        samples_prob[:,0] = torch.ones(output.size())

        # Pass through decoder and sample from resulting vocab distribution
        for t in range(1, self.max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)

            hiddens.append(hidden)

            # Sample token for entire batch from predicted vocab distribution
            batch_token_sample = torch.multinomial(torch.exp(output), 1).view(-1).data
            prob = torch.exp(output).gather(1, batch_token_sample.unsqueeze(1)).view(-1).data
            samples_prob[:, t] = prob
            samples[:, t] = batch_token_sample
            output = autograd.Variable(batch_token_sample)

        return samples, samples_prob, hiddens


    def monte_carlo(self, dis, context, seq, hiddens, num_samples):

        """
        Samples the network using a batch of source input sequence. Passes these inputs
        through the decoder and instead of taking the top1 (like in forward), sample
        using the distribution over the vocabulary


        Inputs: start of sequence, maximum sample sequence length and num of samples
        Outputs: samples
        samples: num_samples x max_seq_length (a sampled sequence in each row)

        Inputs: dialogue context (and maximum sample sequence length
        Outputs: samples
            - samples: batch_size x reply_length x num_samples x max_seq_length"""

        # Initialize sample
        batch_size = seq.size(0)
        vocab_size = self.decoder.output_size
        samples = autograd.Variable(torch.zeros(batch_size, self.max_len))
        samples_prob = autograd.Variable(torch.zeros(batch_size, self.max_len))
        encoder_output, _ = self.encoder(context.permute(1,0))
        rewards = torch.zeros(self.max_len, num_samples, batch_size)
        for t in range(seq.size(1)):

            hidden = hiddens[t]     # Hidden state from orignal generated sequence until t
            output = autograd.Variable(seq[:,t])

            for i in range(num_samples):

                samples_prob[:,0] = torch.ones(output.size())

                # Pass through decoder and sample from resulting vocab distribution
                for next_t in range(t+1, self.max_len):

                    output, hidden, attn_weights = self.decoder(
                            output.long(), hidden, encoder_output)

                    # Sample token for entire batch from predicted vocab distribution
                    batch_token_sample = torch.multinomial(torch.exp(output), 1).view(-1).data
                    prob = torch.exp(output).gather(1, batch_token_sample.unsqueeze(1)).view(-1).data
                    samples_prob[:, next_t] = prob
                    samples[:, next_t] = batch_token_sample
                    output = autograd.Variable(batch_token_sample)
                reward = dis.batchClassify(context.long(), samples.long())
                rewards[t, i, :] = reward
        reward_per_word = torch.mean(rewards, dim=1).permute(1, 0)
        return reward_per_word


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

    def batchPGLoss(self, inp, target, reward, word_probabilites, lamb=0, MC_LM=False):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)
            - Lambda: If causal-entropy lambda > 0

            inp should be target with <s> (start letter) prepended
        """

        batch_size, max_len = target.shape
        loss = 0

        for batch in range(batch_size):
            for word in range(max_len - 1): # No end of sequence token
                if MC_LM:
                    ### KLOPT NIET
                    #  \pi(a|s) --> p(Word|State, CONTEXT)  Reward for word k --> CE(word|state)
                    loss += word_probabilites[batch][word] * reward[batch][word] # LOG PROBALITIES ?? FIX
                else:
                    # Sentence level reward
                    loss += word_probabilites[batch][word] * reward[batch] # LOG PROBALITIES ?? FIX

        # loss = 0
        # for i in range(seq_len):
        #     out, h = self.forward(inp[i], h)
        #     # TODO: should h be detached from graph (.detach())?
        #     for j in range(batch_size):
        #         loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q
        loss = - torch.mean(loss)
        if lamb > 0:
            loss = loss - lamb * torch.mean(- word_probabilites.log()) # CAUSAL ENTROP --> NOT SURE IF IT WORKS THIS WAY
        return loss
