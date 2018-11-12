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

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

import generator
import discriminator
import helpers
from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA = False
VOCAB_SIZE = 5000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 30
START_LETTER = 0
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

def train_generator_MLE(gen, optimizer, data, epochs):
    # Max Likelihood Pretraining for the generator
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for (i, (context, reply)) in enumerate(train_data_loader):
            optimizer.zero_grad()
            context = context.permute(1,0)
            reply = reply.permute(1,0)
            output = gen.forward(context, reply)

            # Compute loss
            pred_dist = output[1:].view(-1, VOCAB_SIZE)
            tgt_tokens = reply[1:].contiguous().view(-1)
            loss = F.nll_loss(pred_dist, tgt_tokens)

            # Backpropagate loss
            loss.backward()
            clip_grad_norm_(gen.parameters(), 10)
            optimizer.step()
            total_loss += loss.data.item()

            # Print updates
            if i % 50 == 0 and i != 0:
                print('[Epoch {} batch {}] loss: {}'.format(total_loss//50))
                total_loss = 0

def train_generator_PG(context, reply, gen, gen_opt, dis):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for one batch.
    """

    # Forward pass
    reply, word_probabilities = gen.sample(context.permute(1,0), MAX_SEQ_LEN)
    rewards = dis.batchClassify(context.long(), reply.long())
    # Backward pass
    gen_opt.zero_grad()
    pg_loss = gen.batchPGLoss(context, reply, rewards, word_probabilities) # FIX
    pg_loss.backward()
    gen_opt.step()


def train_discriminator(context, real_reply, discriminator, dis_opt, generator):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # Batchsize is 32
    # context is 32 x max_context_size

    fake_reply = gen.samples(context, MAX_SEQ_LEN)
    fake_targets = torch.zeros(BATCH_SIZE)
    real_targets = torch.ones(BATCH_SIZE)

    replies = torch.cat((fake_reply, real_reply), 0) # 2x Batchsize
    targets = torch.cat((fake_targets, real_targets), 0)
    context = torch.cat((context, context), 0) # For fixing true and false data
    dis_opt.zero_grad()
    out = discriminator.batchClassify(context, replies)
    loss_fn = nn.BCELoss()
    loss = loss_fn(out, targets)
    loss.backward()
    dis_opt.step()

    total_loss = loss.data.item()
    total_acc = torch.sum((out>0.5)==(targets>0.5)).data.item()/(2 * BATCH_SIZE)

    print(' average_loss = %.4f, train_acc = %.4f' % (
        total_loss, total_acc))

# MAIN
if __name__ == '__main__':
    # Load data set
    if not os.path.isfile("dataset.pickle"):
        print("Saving the data set")


        corpus = DPCorpus(vocabulary_limit=5000)
        train_dataset = corpus.get_train_dataset(min_reply_length=MIN_SEQ_LEN,\
            max_reply_length=MAX_SEQ_LEN)
        train_data_loader = DPDataLoader(train_dataset)
        with open('dataset.pickle', 'wb') as handle:
            pickle.dump(train_data_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading the data set")
        with open('dataset.pickle', 'rb') as handle:
            train_data_loader= pickle.load(handle)

    # Initalize Networks and optimizers
    gen = generator.Generator(VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, device=DEVICE)
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)


    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis_optimizer = optim.Adagrad(dis.parameters()) ## ADAGRAD ??


    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()

    # OPTIONAL: Pretrain generator
    print('Starting Generator MLE Training...')
    # train_generator_MLE(gen, gen_optimizer, train_data_loader, MLE_TRAIN_EPOCHS)

    # #  OPTIONAL: Pretrain discriminator
    # print('\nStarting Discriminator Training...')
    # train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 50, 3)

    # # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        sys.stdout.flush()
        for (batch, (context, reply)) in enumerate(train_data_loader):

            train_generator_PG(context, reply, gen, gen_optimizer, dis)

            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            # train_discriminator(context, reply, dis, dis_optimizer, gen)
