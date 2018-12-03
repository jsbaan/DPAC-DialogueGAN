import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import sys

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, device='cpu', dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru_context = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru_reply = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        return h.to(self.device)

    def forward(self, context, reply, hidden_context, hidden_reply):
        # input dim                                                         # batch_size x seq_len

        emb_context = self.embeddings(context)                              # batch_size x seq_len x embedding_dim
        emb_context = emb_context.permute(1, 0, 2)                          # seq_len x batch_size x embedding_dim
        _, hidden_context = self.gru_context(emb_context, hidden_context)   # 4 x batch_size x hidden_dim
        hidden_context = hidden_context.permute(1, 0, 2).contiguous()       # batch_size x 4 x hidden_dim

        emb_reply = self.embeddings(reply)
        emb_reply = emb_reply.permute(1, 0, 2)
        _, hidden_reply = self.gru_reply(emb_reply, hidden_reply)
        hidden_reply = hidden_reply.permute(1, 0, 2).contiguous()

        # concatenate context with reply
        hidden = torch.cat((hidden_context, hidden_reply), dim=2)

        out = self.gru2hidden(hidden.view(-1, 2*4*self.hidden_dim))             # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                          # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, context, reply):
        """
        Classifies a batch of sequences
.
        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h_context = self.init_hidden(context.size()[0])
        h_reply = self.init_hidden(reply.size()[0])
        out = self.forward(context, reply, h_context, h_reply)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)


