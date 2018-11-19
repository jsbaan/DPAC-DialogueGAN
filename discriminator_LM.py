import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru_response = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=False, dropout=dropout)
        self.gru2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, context, response, hidden_context, hidden_response):
        # input dim                                                         # batch_size x seq_len
        # batch_size x 4 x hidden_dim
        emb_response = self.embeddings(response)
        emb_response = emb_response.permute(1, 0, 2)
        _, hidden_response = self.gru_response(emb_response, hidden_response)
        hidden_response = hidden_response.permute(1, 0, 2).contiguous()


        out = self.gru2hidden(hidden_response)             # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                          # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, context, response):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h_context = self.init_hidden(context.size()[0])
        h_response = self.init_hidden(response.size()[0])
        out = self.forward(context, response, h_context, h_response)
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


