from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
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
MAX_SEQ_LEN = 60
START_LETTER = 0
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'


def train_generator_MLE(gen, optimizer, data, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for (i, (context, reply)) in enumerate(train_data_loader):
            optimizer.zero_grad()
            context = torch.tensor(context).permute(1,0)
            reply = torch.tensor(reply).permute(1,0)
            output = gen(context, reply)

            # Compute loss
            pred_dist = output[1:].view(-1, VOCAB_SIZE)
            tgt_tokens = reply[1:].contiguous().view(-1)
            loss = F.nll_loss(pred_dist, tgt_tokens)

            # Backpropagate loss
            loss.backward()
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
    reply = gen(context, reply, teacher_forcing_ratio=0)
    _, reply = torch.max(reply, dim=2)
    print(context.shape)
    print(reply.shape)
    rewards = dis.batchClassify(context, reply.permute(1,0))

    # Backward pass
    gen_opt.zero_grad()
    pg_loss = gen.batchPGLoss(context, reply, rewards) # FIX
    pg_loss.backward()
    gen_opt.step()


def train_discriminator(context, real_reply, discriminator, dis_opt, generator):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # Batchsize is 32
    # context is 32 x max_context_size

    # fake_reply = gen.samples(context)
    fake_reply = real_reply ## TEMPORARY FIX
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
        corpus = DPCorpus(vocabulary_limit=5000, batch_size=BATCH_SIZE)
        train_dataset = corpus.get_train_dataset()
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
    # quit()

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
