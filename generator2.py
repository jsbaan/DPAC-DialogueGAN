import torch
import torch.nn as nn
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNN import DecoderRNN
from seq2seq.TopKDecoder import TopKDecoder
from seq2seq.Seq2Seq import Seq2seq

class Generator2(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            embed_size,
            max_len,
            beam_size=5,
            enc_n_layers=2,
            enc_dropout=0.2,
            dec_n_layers=2,
            dec_dropout=0.2,
            teacher_forcing_ratio=0.5):
        super(Generator2, self).__init__()

        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = EncoderRNN(vocab_size, max_len-1, hidden_size, 0, enc_dropout, enc_n_layers, True, 'gru', False, None)
        self.decoder = DecoderRNN(vocab_size, max_len-1, hidden_size * 2, 2, 5, dec_n_layers, 'gru', True, 0, dec_dropout, True)
        self.beam_decoder = TopKDecoder(self.decoder, beam_size)
        self.seq2seq = Seq2seq(self.encoder, self.beam_decoder)

    def forward(self, src, tgt):
        outputs, _, _ = self.seq2seq(src, target_variable=tgt, teacher_forcing_ratio=self.teacher_forcing_ratio)

        start_tokens = torch.zeros(64, self.vocab_size, device=outputs[0].device)
        start_tokens[5,:] = 1

        outputs = [start_tokens] + outputs
        outputs = torch.stack(outputs)
        return outputs