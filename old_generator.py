import torch
import torch.nn as nn
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNN import DecoderRNN
from seq2seq.TopKDecoder import TopKDecoder
from seq2seq.Seq2Seq import Seq2seq
import sys
class Generator(nn.Module):
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
        super(Generator, self).__init__()

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

    def PG_loss(self, reward, word_probabilites, lamb=0):
        sentence_level_reward = torch.mean(reward, 1)
        if lamb > 0:
            loss = loss - lamb * torch.mean(- word_probabilites.log()) # CAUSAL ENTROP --> NOT SURE IF IT WORKS THIS WAY
        return loss
