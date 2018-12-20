import torch.nn as nn
from torch.autograd import Variable
import torch

class LM(nn.Module):
  """Simple LSMT-based language model"""
  def __init__(self, embedding_dim, num_steps, batch_size, vocab_size, num_layers):
    super(LM, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(0.2)
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=num_layers)
    self.sm_fc = nn.Linear(in_features=embedding_dim,
                           out_features=vocab_size)
    self.init_weights()

  def init_weights(self):
    init_range = 0.1
    self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()),
            Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()))

  def forward(self, inputs):
    embeds = self.dropout(self.word_embeddings(inputs))
    lstm_out, hidden = self.lstm(embeds)
    lstm_out = self.dropout(lstm_out)
    logits = torch.softmax(self.sm_fc(lstm_out.view(-1, self.embedding_dim)), 1)
    return logits.view(self.batch_size, self.num_steps, self.vocab_size), hidden

  def get_rewards(self, reply, PAD):
    probabilities, _ = self.forward(reply.long())
    rewards = torch.zeros(self.batch_size, self.num_steps - 1)
    indices = torch.arange(self.batch_size).long()
    if torch.cuda.is_available():
      indices = indices.cuda()
      rewards = rewards.cuda()

    for t in range(self.num_steps - 1):
      rewards[indices, t] = probabilities[indices, t, reply[:, t+1].view(-1)].view(-1)
    padding_matrix = (reply != PAD).float()
    sum_paddin_matrix = torch.sum(padding_matrix, 1)
    rewards = rewards * padding_matrix[:, 1:]
    sentence_level_rewards = torch.sum(rewards, 1)/sum_paddin_matrix
    return rewards, sentence_level_rewards


def repackage_hidden(h):
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)