import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class Critic(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, dropout=0.2, device="cpu"):
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru_response = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=False, dropout=dropout)
        self.gru2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*1, batch_size, self.hidden_dim))

        if self.device != "cpu":
            return h.cuda()
        else:
            return h

    def forward(self, state, hidden=None):
        # input dim: batch_size x seq_len
        # batch_size x 4 x hidden_dim
        emb_response = self.embeddings(state) # batchsize x embedding dim
        emb_response = emb_response.permute(1, 0, 2)
        _, hidden_response = self.gru_response(emb_response, hidden)
        hidden_response = hidden_response.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden_response[:, -1, :]) # batch_size x 4*hidden_dim
        out = torch.relu(out)
        q_values = self.hidden2out(out) # batch_size x 1
        return q_values

    def batchClassify(self, state):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h_state = self.init_hidden(response.size()[0])
        q_values = self.forward(state, h_state)
        return q_values

    def batchBCELoss(self, state, next_state, target, done, discount_factor):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        value_function = self.forward(state)
        with torch.no_grad():
            value_function_next = self.forward(next_state)
            mask = (1 - done).float()
            target = mask * (reward +  (discount_factor * value_function_next))
        loss = F.smooth_l1_loss(value_function, target)
        return loss
