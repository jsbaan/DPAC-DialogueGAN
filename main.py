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
MAX_SEQ_LEN = 20
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 2
ADV_TRAIN_EPOCHS = 50

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

def train_generator_MLE(gen, optimizer, data, epochs):
    # Max Likelihood Pretraining for the generator
    pad_token = data.dataset.corpus.token_to_id('<pad>')
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        losses = []
        for (iter, (context, reply)) in enumerate(train_data_loader):
            print('Epoch {} Iter {}'.format(epoch+1,iter))
            optimizer.zero_grad()
            context = context.permute(1,0)
            reply = reply.permute(1,0)
            output = gen.forward(context, reply)

            # Compute loss
            pred_dist = output[1:].view(-1, VOCAB_SIZE)
            tgt_tokens = reply[1:].contiguous().view(-1)

            loss = F.nll_loss(pred_dist, tgt_tokens, ignore_index=pad_token)

            # Backpropagate loss
            loss.backward()
            clip_grad_norm_(gen.parameters(), 10)
            optimizer.step()
            total_loss += loss.data.item()
            losses.append(loss)

            # Print updates
            if iter % 50 == 0 and iter != 0:
                print('[Epoch {} iter {}] loss: {}'.format(epoch,iter,total_loss//50))
                total_loss = 0
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': gen.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss'      : losses,
                },'generator_checkpoint.pth.tar')
    return losses

def train_generator_PG(context, reply, gen, gen_opt, dis):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for one batch.
    """

    # Forward pass
    reply, word_probabilities = gen.sample(context.permute(1,0), MAX_SEQ_LEN)
    entropy = torch.mean(word_probabilities.log(), dim=1)
    perplexity = torch.mean(2**(-entropy)).item()

    rewards = dis.batchClassify(context.long(), reply.long())
    # Backward pass
    gen_opt.zero_grad()
    pg_loss = gen.batchPGLoss(context, reply, rewards, word_probabilities) # FIX
    pg_loss.backward()
    gen_opt.step()
    return perplexity


def train_discriminator(context, real_reply, discriminator, dis_opt, generator, corpus):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # Batchsize is 32
    # context is 32 x max_context_size

    fake_reply, _ = gen.sample(context.permute(1,0), MAX_SEQ_LEN)

    # UNCOMMENT FOR PRINTING SAMPLES AND CONTEXT

    # print(corpus.ids_to_tokens([int(i) for i in context[0]]))
    # print("Fake generated reply")
    # print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
    # print("Real  reply")
    # print(corpus.ids_to_tokens([int(i) for i in real_reply[0]]))
    # print(30 * "-")

    fake_targets = torch.zeros(BATCH_SIZE)
    real_targets = torch.ones(BATCH_SIZE)

    # replies = torch.cat((fake_reply.long(), real_reply), 0) # 2x Batchsize
    # targets = torch.cat((fake_targets, real_targets), 0)
    # context = torch.cat((context, context), 0) # For fixing true and false data

    dis_opt.zero_grad()
    out_fake = discriminator.batchClassify(context, fake_reply.long())
    out_real = discriminator.batchClassify(context, real_reply.long())



    loss_fn = nn.BCELoss()
    loss_fake = loss_fn(out_fake, fake_targets)

    loss_real = loss_fn(out_real, real_targets)


    loss = loss_real + loss_fake
    loss.backward()
    dis_opt.step()

    total_loss = loss.data.item()
    out = torch.cat((out_fake, out_real), 0)
    targets = torch.cat((real_targets, fake_targets), 0)
    correct_real = torch.sum(out_real > 0.5)/BATCH_SIZE
    correct_fake = torch.sum(out_fake < 0.5)/BATCH_SIZE
    total_acc = (correct_real + correct_fake)/2
    print(' average_loss = %.4f, train_acc = %.4f' % (
        total_loss, total_acc))

# MAIN
if __name__ == '__main__':
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
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)


    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis_optimizer = optim.Adagrad(dis.parameters()) ## ADAGRAD ??


    if CUDA:
        dis = dis.cuda()

    # OPTIONAL: Pretrain generator
    # checkpoint = torch.load('generator_checkpoint.pth.tar')
    print('Starting Generator MLE Training...')
    train_generator_MLE(gen, gen_optimizer, train_data_loader, MLE_TRAIN_EPOCHS)

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
            print('\nAdversarial Training Generator: ')
            perplexity = train_generator_PG(context, reply, gen, gen_optimizer, dis)
            if batch % 10 == 0:
                print("After " + str(batch) + " batches, the perplexity is: " + str(perplexity))

            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            train_discriminator(context, reply, dis, dis_optimizer, gen, corpus)
