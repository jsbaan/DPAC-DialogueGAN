from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

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
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
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

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            optimizer.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


def train_generator_PG(context, gen, gen_opt, dis):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for one batch.
    """

    # Forward pass
    reply = gen(context, _, 0)
    print(reply)
    rewards = dis.batchClassify(context, reply)
    
    # Backward pass 
    gen_opt.zero_grad()
    pg_loss = gen.batchPGLoss(inp, target, rewards) # FIX
    pg_loss.backward()
    gen_opt.step()


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

# MAIN
if __name__ == '__main__':
    # Load data set
    if not os.path.isfile("dataset.pickle"):
        print("Saving the data set")
        corpus = DPCorpus(vocabulary_limit=5000)
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

    # dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    # dis_optimizer = optim.Adagrad(dis.parameters()) ## ADAGRAD ??

    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()

    # OPTIONAL: Pretrain generator
    print('Starting Generator MLE Training...')
    train_generator_MLE(gen, gen_optimizer, train_data_loader, MLE_TRAIN_EPOCHS)
    quit()

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
            train_generator_PG(context, gen, gen_optimizer, dis)
            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            train_discriminator(dis, dis_optimizer, gen, 5, 3)
