import torch
import torch.nn as nn
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNN import DecoderRNN
from seq2seq.TopKDecoder import TopKDecoder
from seq2seq.Seq2Seq import Seq2seq
import sys
import time
from torch.nn.utils import clip_grad_norm_

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

        batch_size = outputs[0].size(0)
        start_tokens = torch.zeros(batch_size, self.vocab_size, device=outputs[0].device)
        start_tokens[:,self.sos_id] = 1

        outputs = [start_tokens] + outputs
        outputs = torch.stack(outputs)
        return outputs

        # NOTICE THAT DISCOUNT FACTOR is 1
    def compute_reinforce_loss(self, rewards, probabilities):
        sentence_level_reward = torch.mean(rewards, 1).unsqueeze(1)
        R_s_w = torch.mul(rewards, sentence_level_reward)

        sent_len = rewards.size(1)
        J = 0
        for k in range(sent_len):
            R_k = torch.sum(R_s_w[:,k:], 1)
            prob = probabilities[:,k+1].log()
            J += R_k*prob

        loss = -torch.mean(J)
        return loss

    def try_get_state_dicts(self,directory='./', prefix='generator_checkpoint', postfix='.pth.tar'):
        files = os.listdir(directory)
        files = [f for f in files if f.startswith(prefix)]
        files = [f for f in files if f.endswith(postfix)]

        epoch_nums = []
        for file in files:
            number = file[len(prefix):-len(postfix)]
            try:
                epoch_nums.append(int(number))
            except:
                pass

        if len(epoch_nums) < 2:
            return None

        last_complete_epoch = sorted(epoch_nums)[-2]
        filename = prefix + str(last_complete_epoch) + postfix

        data = torch.load(filename)
        return data

    def train_generator_MLE_batch(self, context, reply, optimizer, pad_id):
        context = context.t()
        reply = reply.t()
        loss_func = torch.nn.NLLLoss(ignore_index=pad_id) # TO DEVICE?
        output = self.forward(context, reply)
        pred_dist = output[1:].view(-1, self.vocab_size)
        tgt_tokens = reply[1:].contiguous().view(-1)
        loss = loss_func(pred_dist, tgt_tokens)

        # Backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10) # might be something to check
        optimizer.step()

    def train_generator_MLE(self, optimizer, data_loader, epochs):
        # Max Likelihood Pretraining for the generator
        corpus = data_loader.dataset.corpus
        pad_id = corpus.token_to_id(corpus.PAD)

        loss_func = torch.nn.NLLLoss(ignore_index=pad_id)

        start_epoch = 0
        # saved_data = try_get_state_dicts()
        # if saved_data is not None:
        #     start_epoch = saved_data['epoch']
        #     self.load_state_dict(saved_data['state_dict'])
        #     optimizer.load_state_dict(saved_data['optimizer'])

        loss_per_epoch = []
        for epoch in range(start_epoch, epochs):
            print('epoch %d : ' % (epoch + 1))

            total_loss = 0
            losses = []
            for (iter, (context, reply)) in enumerate(data_loader):
                optimizer.zero_grad()
                context = context.t()
                reply = reply.t()
                output = self.forward(context, reply)

                # Compute loss
                pred_dist = output[1:].view(-1, self.vocab_size)
                tgt_tokens = reply[1:].contiguous().view(-1)

                loss = loss_func(pred_dist, tgt_tokens)

                # Backpropagate loss
                loss.backward()
                clip_grad_norm_(self.parameters(), 10)
                optimizer.step()
                total_loss += loss.data.item()
                losses.append(loss)

                # Print updates
                if iter % 50 == 0 and iter != 0:
                    print('[Epoch {} iter {}] loss: {}'.format(epoch,iter,total_loss/50))
                    total_loss = 0
                    torch.save({
                        'epoch': epoch+1,
                        'state_dict': self.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'loss'      : losses,
                    },'generator_checkpoint{}.pth.tar'.format(epoch))

                    try:
                        print("Generated reply")
                        print(' '.join(corpus.ids_to_tokens([int(i) for i in output.argmax(2)[:,0]])))
                        print("Real  reply")
                        print(' '.join(corpus.ids_to_tokens([int(i) for i in reply[:,0]])))
                    except:
                        print("Unable to print")

            loss_per_epoch.append(total_loss)
        torch.save(loss_per_epoch, "generator_final_loss.pth.tar")
        return losses
