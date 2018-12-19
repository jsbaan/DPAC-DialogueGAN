import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import time

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2, device='cpu'):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.device = device

        ## Reply embedding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)

        # context embedding
        self.embeddings2 = nn.Embedding(vocab_size, embedding_dim)
        self.gru2 = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden2 = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear2 = nn.Dropout(p=dropout)

        self.hidden2out = nn.Linear(2 * hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim)).to(self.device)
        return h

    def forward(self, reply, context, hidden, hidden2):
        # REPLy dim                                                # batch_size x seq_len
        emb = self.embeddings(reply)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out_reply = self.dropout_linear(out)

        # Context
        emb = self.embeddings2(context)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru2(emb, hidden2)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden2(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out_context = self.dropout_linear2(out)
        
        out = self.hidden2out(torch.cat((out_reply, out_context), 1))  # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, reply, context):
        """
        Classifies a batch of sequences.
        Inputs: inp
            - inp: batch_size x seq_len
        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(reply.size()[0])
        h2 = self.init_hidden(context.size()[0])
        out = self.forward(reply.long(), context.long(), h, h2)
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