import torch.nn as nn
from torch.autograd import Variable
import torch

class LM(nn.Module):
    """Simple LSMT-based language model"""
    def __init__(self, embedding_dim, vocab_size, device='cpu'):
        super(LM, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim,
                                hidden_size=128,
                                num_layers=2)
        self.Linear1 = nn.Linear(in_features=128,
                               out_features=500)
        self.Linear2 = nn.Linear(in_features=500,
                               out_features=vocab_size)


    def forward(self, reply):
        reply = reply.long()
        embed = self.word_embeddings(reply).permute(1, 0, 2)
        out, hidden = self.gru(embed)
        out = torch.relu(out.permute(1, 2, 0)[:, :, -1])
        out = torch.relu(self.Linear1(out))
        out = torch.softmax(self.Linear2(out), 1)
        return out

    def get_rewards(self, reply, PAD):
        reply = reply.long().to(self.device)
        reward_length = reply.size(1) - 1
        rewards = torch.zeros(reply.size(0), reward_length).to(self.device)
        indices = torch.arange(reply.size(0)).to(self.device)
        for t in range(reward_length):
            input_t = reply[:, :t+1]
            label = reply[:, t+1]
            prediction = self.forward(input_t)
            rewards[indices, t] = torch.log(prediction[indices, label.view(-1)] + 1e-6)
        padding_matrix = (reply[:, 1:] != PAD).float()
        padding_matrix_sum = torch.sum(padding_matrix, 1) # count number of non padding numbers for average
        rewards = (rewards * padding_matrix) - (reply[:, 1:] == PAD).float() * 7
        sentence_level_reward = torch.sum(rewards, 1)/padding_matrix_sum
        return rewards, sentence_level_reward

    def get_reward(self, state, action):
        state = state.long().to(self.device)
        indices = torch.arange(state.size(0)).to(self.device)
        prediction = self.forward(state)
        reward = torch.log(prediction[indices, action.view(-1)] + 1e-6)
        return reward
