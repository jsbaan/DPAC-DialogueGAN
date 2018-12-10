# TODO
    # Strategies that improve response diversity (https://arxiv.org/pdf/1701.06547.pdf)
        # 1) Instead of using
        # the same learning rate for all examples, using a
        # weighted learning rate that considers the average
        # tf-idf score for tokens within the response. Such
        # a strategy decreases the influence from dull and
        # generic utterances
        # 2) Penalizing word types (stop words
        # excluded) that have already been generated. Such
        # a strategy dramatically decreases the rate of repetitive
        # responses such as no. no. no. no. no. or contradictory
        # responses such as I don’t like oranges
        # but i like oranges.

from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
# import nltk
# nltk.download('punkt')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

import generator
import discriminator
import discriminator_LM
from helpers import *
from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os
import time

DEVICE = torch.device('cuda:0')  #'cpu'
VOCAB_SIZE = 8000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 2
ADV_TRAIN_EPOCHS = 100
DIS_TRAIN_EPOCHS = 100


GEN_EMBEDDING_DIM = 256
GEN_HIDDEN_DIM = 256
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64
DISCRIMINATOR_LM = True     # one of the two (DISCRIMINATOR_LM or MC) must be False
MC = False

def try_get_state_dicts(directory='./', prefix='generator_checkpoint', postfix='.pth.tar'):
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


def fill_with_padding(sentences, u_token, pad_token):
    """
    Takes a batch of sentences with equal lengths as input. 
    Returns same size of batch but with padding filling after the first 
    end of utterence token.
    """

    for i in range(sentences.size(0)):
        sent = sentences[i]
        idx = (sent == u_token).nonzero()
        if len(idx) > 0:    
            idx = idx[0].item()
            split = torch.split(sent, idx+1)[0]
            padding = pad_token * torch.ones(sentences.size(1) - len(split))
            padding = padding.to(DEVICE)
            pad_sent = torch.cat((split, padding))
            sentences[i][:] = pad_sent

    return sentences


def train_discriminator(discriminator, dis_opt, generator, corpus, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # Batchsize is 32
    # context is 32 x max_context_size

    ignore_index = corpus.token_to_id('<pad>')
    ud_id = corpus.token_to_id('</u>')

    start_epoch = 0
    # saved_dis = try_get_state_dicts(prefix='discriminator_checkpoint') 
    # if saved_dis is not None:
    #     start_epoch = saved_dis['epoch']
    #     discriminator.load_state_dict(saved_dis['state_dict'])
    #     dis_opt.load_state_dict(saved_dis['optimizer'])

    loss_per_epoch = []

    for epoch in range(start_epoch, epochs):
        print('epoch %d : ' % (epoch + 1))

        total_loss = 0
        losses = []

        for (iter, (context, real_reply)) in enumerate(train_data_loader):
            dis_opt.zero_grad()

            fake_reply, _, _ = gen.sample(context.permute(1,0), MAX_SEQ_LEN)
            fake_reply = fill_with_padding(fake_reply, ud_id, ignore_index)

            if DISCRIMINATOR_LM:

                fake_rewards = torch.sum(discriminator.get_rewards(fake_reply, ignore_index), dim=1)
                real_rewards = torch.sum(discriminator.get_rewards(real_reply, ignore_index), dim=1)

                loss = -torch.mean((real_rewards - fake_rewards))

                # print("Fake generated reply")
                # print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
                # print("Real  reply")
                # print(corpus.ids_to_tokens([int(i) for i in real_reply[0]]))

                # print("fake reward ", torch.mean(fake_rewards).item())
                # print("real reward ", torch.mean(real_rewards).item())


            else:
                fake_targets = torch.zeros(BATCH_SIZE)
                real_targets = torch.ones(BATCH_SIZE)

                dis_opt.zero_grad()
                out_fake = discriminator.batchClassify(context, fake_reply.long())
                out_real = discriminator.batchClassify(context, real_reply.long())

                loss_fn = nn.BCELoss()
                loss_fake = loss_fn(out_fake, fake_targets)

                loss_real = loss_fn(out_real, real_targets)

                loss = loss_real + loss_fake
                total_loss = loss.data.item()
                out = torch.cat((out_fake, out_real), 0)
                targets = torch.cat((real_targets, fake_targets), 0)
                correct_real = torch.sum(out_real > 0.5)/BATCH_SIZE
                correct_fake = torch.sum(out_fake < 0.5)/BATCH_SIZE
                total_acc = (correct_real + correct_fake)/2
                print(' average_loss = %.4f, train_acc = %.4f' % (
                    total_loss, total_acc))

            loss.backward()
            dis_opt.step()
            total_loss += loss.data.item()
            losses.append(loss)

            # print updates
            # if iter % 50 == 0 and iter != 0:
            if iter % 50 == 0:
                print('[Epoch {} iter {}] loss: {}'.format(epoch,iter,total_loss/50))
                total_loss = 0
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': discriminator.state_dict(),
                    'optimizer' : dis_opt.state_dict(),
                    'loss'      : losses,
                },'discriminator_checkpoint{}.pth.tar'.format(epoch))

                try:
                    print("Fake generated reply")
                    print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
                    print("Real  reply")
                    print(corpus.ids_to_tokens([int(i) for i in real_reply[0]]))

                    print("fake reward ", torch.mean(fake_rewards).item())
                    print("real reward ", torch.mean(real_rewards).item())
                except:
                    print("Unable to print")

        loss_per_epoch.append(total_loss)
    torch.save(loss_per_epoch, "discriminator_final_loss.pth.tar")





# MAIN
if __name__ == '__main__':

    # Do we have enough arguments?
    assert len(sys.argv) == 2, "You should pass file name of pretrained generator as argument" 

    # Load data set
    if not os.path.isfile("dataset.pickle"):
        print("Saving the data set")


        corpus = DPCorpus(vocabulary_limit=VOCAB_SIZE)
        train_dataset = corpus.get_train_dataset(min_reply_length=MIN_SEQ_LEN,\
            max_reply_length=MAX_SEQ_LEN)
        train_data_loader = DPDataLoader(train_dataset,batch_size=BATCH_SIZE)
        with open('dataset.pickle', 'wb') as handle:
            pickle.dump(train_data_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            corpus = train_data_loader.dataset.corpus
    else:
        print("Loading the data set")
        with open('dataset.pickle', 'rb') as handle:
            train_data_loader= pickle.load(handle)
        corpus = train_data_loader.dataset.corpus


    # Initalize Networks and optimizers
    gen = generator.Generator(VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, device=DEVICE)
    # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)

    # initialize generator
    # saved_data = try_get_state_dicts()
    # gen.load_state_dict(saved_data['state_dict'])

    saved_data = torch.load(sys.argv[1])
    gen.load_state_dict(saved_data['state_dict'])

    if DISCRIMINATOR_LM:
        dis = discriminator_LM.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)
    else:
        dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)


    dis = dis.to(DEVICE)
    dis_optimizer = optim.Adagrad(dis.parameters()) ## ADAGRAD ??


    #  OPTIONAL: Pretrain discriminator
    print('\nStarting Discriminator Training...')
    train_discriminator(dis, dis_optimizer, gen, corpus, DIS_TRAIN_EPOCHS)





