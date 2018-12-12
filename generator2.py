import torch
import torch.nn as nn
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNN import DecoderRNN
from seq2seq.TopKDecoder import TopKDecoder
from seq2seq.Seq2Seq import Seq2seq
import sys
class Generator2(nn.Module):
    def __init__(
            self,
            sos_id,
            eou_id,
            vocab_size,
            hidden_size,
            embed_size,
            max_len,
            beam_size=3,
            enc_n_layers=2,
            enc_dropout=0.2,
            enc_bidirectional=True,
            dec_n_layers=2,
            dec_dropout=0.2,
            dec_bidirectional=True,
            teacher_forcing_ratio=0.5):
        super(Generator2, self).__init__()

        self.sos_id = sos_id
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = EncoderRNN(vocab_size, max_len-1, hidden_size, 0, enc_dropout, enc_n_layers, True, 'gru', False, None)
        self.decoder = DecoderRNN(vocab_size, max_len-1, hidden_size*2 if dec_bidirectional else hidden_size, sos_id, eou_id, dec_n_layers, 'gru', dec_bidirectional, 0, dec_dropout, True)
        # self.beam_decoder = TopKDecoder(self.decoder, beam_size)
        self.seq2seq = Seq2seq(self.encoder, self.decoder)

    def sample(self, src, tgt):
        sentences, probabilities = self.seq2seq(src, target_variable=tgt, teacher_forcing_ratio=0, sample=True)
        return sentences, probabilities

    def forward(self, src, tgt):
        src = src.t()
        tgt = tgt.t()
        outputs, _, _ = self.seq2seq(src, target_variable=tgt, teacher_forcing_ratio=self.teacher_forcing_ratio)
        start_tokens = torch.zeros(64, self.vocab_size, device=outputs[0].device)
        start_tokens[:,self.sos_id] = 1

        outputs = [start_tokens] + outputs
        outputs = torch.stack(outputs)
        return outputs

    def compute_reinforce_loss(self, rewards, probabilities):
        sentence_level_reward = torch.mean(rewards, 1).unsqueeze(1)
        rewards_sentence = torch.mul(rewards, sentence_level_reward)
        returns = torch.stack([probabilities[:, i+1].log() * torch.sum(rewards_sentence[:, i:], 1)\
         for i in range(rewards.size(1))]).t()
        returns_sum = torch.sum(returns, 1)
        loss = -torch.mean(returns_sum)
        return loss