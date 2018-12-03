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
    # Hierarchical decoder to generate multiple sentences

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
import discriminator_LM
import critic

from helpers import *
from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os
import time
import replay_memory

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')  #'
else:
    DEVICE = torch.device('cpu')  #'cuda:0'

VOCAB_SIZE = 5000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 50
ADV_TRAIN_EPOCHS = 50
LM_TRAIN_EPOCHS = 20

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64
MC = False
USE_EXP_REPLAY = False
CAPACITY_RM = 100000
PRETRAIN = False
DISCRIMINATOR_LM = True     # one of the two (DISCRIMINATOR_LM or MC) must be False
AC_WARMUP = 1000

def train_generator_MLE(gen, optimizer, data, epochs):
    # Max Likelihood Pretraining for the generator
    pad_token = data.dataset.corpus.token_to_id('<pad>')
    loss_per_epoch = []
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
                },'generator_checkpoint{}.pth.tar'.format(epoch))
        loss_per_epoch.append(total_loss)
    torch.save(loss_per_epoch, "generator_final_loss.pth.tar")
    return losses

def train_generator_PG(context, reply, gen, gen_opt, dis):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for one batch.
    """
    # Forward pass
    reply, word_probabilities, hiddens = gen.sample(context.permute(1,0), MAX_SEQ_LEN)
    entropy = torch.mean(word_probabilities.log(), dim=1)
    perplexity = torch.mean(2**(-entropy)).item()

    if MC:
        rewards = gen.monte_carlo(dis, context, reply, hiddens, num_samples=1)
    elif DISCRIMINATOR_LM:
        rewards = dis.get_rewards(reply)
    else:
        rewards = dis.batchClassify(context.long(), reply.long())

    # Backward pass
    gen_opt.zero_grad()
    if MC or DISCRIMINATOR_LM == True:
        pg_loss = gen.batchPGLoss(context, reply, rewards, word_probabilities, MC_LM=True) # FIX
    else:
        pg_loss = gen.batchPGLoss(context, reply, rewards, word_probabilities, MC_LM=False)

    pg_loss.backward()
    gen_opt.step()
    return perplexity

def train_generator_PGAC(context, reply, gen, gen_opt, dis, memory, critic):
    """
    Actor Critic Pseudocode:

    for word, t in enumerate(setence):
        state = [word_0, ..., word_t]
        action = gen.forward(word)
        next_state = [word_0, ..., word_{t+1}]
        reward = dis(word{t+1} | state)
        store (s, a, r, s', done ) in replay memory

        # Training
        sample batch from replay memory
        Update critic --> r + discount_facot * V(s') - V(s)   NOTE: target with no grad!
        update actor --> torch.mean(V(s)) NOTE: not like policy gradient, but according to Deepmind DDPG

        Question: Could also update discriminator in this loop?
    """
    # Run input through encoder
    encoder_output, hidden = gen.encoder(context)
    hidden = hidden[:gen.decoder.n_layers]
    input = torch.autograd.Variable(context.data[0, :])  # sos
    samples = torch.autograd.Variable(torch.zeros(BATCH_SIZE,MAX_SEQ_LEN)).to(DEVICE)
    samples[:,0] = input

    # Pass through decoder and sample action (word) from resulting vocab distribution
    for t in range(1, MAX_SEQ_LEN):
        output, hidden, attn_weights = gen.decoder(
                input, hidden, encoder_output)

        # Sample action (token) for entire batch from predicted vocab distribution
        action = torch.multinomial(torch.exp(output), 1).view(-1).data
        state = samples[:,:t]
        reward = dis.get_reward(state, action)
        samples[:, t] = action
        next_state = samples[:,:t+1]
        # TODO fix done by checking for terminal states (in batch?how?)
        done = None
        input = torch.autograd.Variable(action)

        ## Train using AC without experience replay
        if not USE_EXP_REPLAY:
            q_values = critic.forward(state)
            with torch.no_grad():
                mask = (done==False).float()
                q_values_target = mask*(discount_factor * critic.forward(next_state)) + reward

            critic_loss = F.smooth_l1_loss(cur_value, next_value)

        ######## LAB CODE FOR ACTOR CRITIC
        # # Compute value/critic loss
        # cur_value = critic(state).squeeze(1)
        # with torch.no_grad():
        #     mask = (done==False).float()
        #     next_value = mask*(discount_factor * critic(next_state).squeeze(1)) + reward
        # value_loss = F.smooth_l1_loss(cur_value, next_value)
        #
        # # Compute actor loss
        # v = reward + discount_factor * critic(next_state).squeeze(1)
        # actor_loss = -torch.mean(log_ps * v.detach())
        #
        #
        # # The loss is composed of the value_loss (for the critic) and the actor_loss
        # loss = value_loss + actor_loss
        #
        # # backpropagation of loss to Neural Network (PyTorch magic)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        #######



            # Update actor (generator) --> torch.mean(V(s)) NOTE: not like policy gradient, but according to Deepmind DDPG
            actor_loss =

        ## Train using experience replay
        elif USE_EXP_REPLAY:
            # TODO decide when done should be true (first padding token generated?)
            # TODO decide whether to put entire batch as one tuple or as seperate entries
            # TODO decide whether padding the experience replay buffer makes sense
            #       (would this solve the issue of variable length states?)

            # Store (batch of) transition(s) in replay memory
            done = False
            memory.push_batch((state,action,reward,next_state,done))
            if memory.__len__() > AC_WARMUP:
                # Sample batch from replay memory
                memory.sample(BATCH_SIZE)
                # Update critic --> r + discount_facot * V(s') - V(s)   NOTE: target with no grad!
                # update actor --> torch.mean(V(s)) NOTE: not like policy gradient, but according to Deepmind DDPG
    quit()
    return

def train_discriminator(context, real_reply, discriminator, dis_opt, generator, corpus):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # Batchsize is 32
    # context is 32 x max_context_size

    # torch.nn.utils.clip_grad_norm(discriminator.parameters(), max_norm=5.0)

    fake_reply, _, _ = gen.sample(context.permute(1,0), MAX_SEQ_LEN)

    # UNCOMMENT FOR PRINTING SAMPLES AND CONTEXT

    # print(corpus.ids_to_tokens([int(i) for i in context[0]]))
    # print("Fake generated reply")
    # print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
    # print("Real  reply")
    # print(corpus.ids_to_tokens([int(i) for i in real_reply[0]]))
    # print(30 * "-")
    if DISCRIMINATOR_LM:
        # print("Generated reply")
        # print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
        # print("Real  reply")
        # print(corpus.ids_to_tokens([int(i) for i in real_reply[0]]))

        fake_rewards = torch.mean(dis.get_rewards(fake_reply), dim=1)
        real_rewards = torch.mean(dis.get_rewards(real_reply), dim=1)
        print("fake reward ", torch.mean(fake_rewards).item())
        print("real reward ", torch.mean(real_rewards).item())
        # print("fake rewards ", fake_rewards)
        # print("real rewards ", real_rewards)
        loss = -torch.mean((real_rewards - fake_rewards))
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

def load_data(path='dataset.pickle'):
    """
    Load data set
    """
    if not os.path.isfile(path):
        print("Saving the data set")
        corpus = DPCorpus(vocabulary_limit=VOCAB_SIZE)
        train_dataset = corpus.get_train_dataset(min_reply_length=MIN_SEQ_LEN,\
            max_reply_length=MAX_SEQ_LEN)
        train_data_loader = DPDataLoader(train_dataset,batch_size=BATCH_SIZE)
        with open(path, 'wb') as handle:
            pickle.dump(train_data_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            corpus = train_data_loader.dataset.corpus
    else:
        print("Loading the data set")
        with open(path, 'rb') as handle:
            train_data_loader= pickle.load(handle)
        corpus = train_data_loader.dataset.corpus
    return corpus,train_data_loader

if __name__ == '__main__':
    '''
    Main training loop. Pre-trains the generator and discriminator using MLE
    and then uses PG to alternately train them.
    '''
    # Load data set
    corpus, train_data_loader = load_data()

    # Initalize Networks and optimizers
    gen = generator.Generator(VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, device=DEVICE)
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)

    if DISCRIMINATOR_LM:
        dis = discriminator_LM.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)
    else:
        dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)


    dis = dis.to(DEVICE)
    dis_optimizer = optim.Adagrad(dis.parameters()) ## ADAGRAD ??

    critic = critic.Critic(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)
    memory = replay_memory.ReplayMemory(CAPACITY_RM)


    # OPTIONAL: Pretrain generator
    if PRETRAIN:
        checkpoint = torch.load('generator_checkpoint.pth.tar')
        print('Starting Generator MLE Training...')
        train_generator_MLE(gen, gen_optimizer, train_data_loader, MLE_TRAIN_EPOCHS)

        #  OPTIONAL: Pretrain discriminator
        print('\nStarting Discriminator Training...')
        for epoch in range(ADV_TRAIN_EPOCHS):
            print('\n--------\nEPOCH %d\n--------' % (epoch+1))
            for (batch, (context, reply)) in enumerate(train_data_loader):
                print('\n Pretraining Discriminator: ')
                train_discriminator(context, reply, dis, dis_optimizer, gen, corpus)


    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        sys.stdout.flush()
        for (batch, (context, reply)) in enumerate(train_data_loader):
            # TRAIN GENERATOR
            print('\nAdversarial Training Generator: ')
            # perplexity = train_generator_PG(context, reply, gen, gen_optimizer, dis)
            perplexity = train_generator_PGAC(context.permute(1,0), reply.permute(1,0),\
                gen, gen_optimizer, dis, memory, critic)
            if batch % 10 == 0:
                print("After " + str(batch) + " batches, the perplexity is: " + str(perplexity))

            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            train_discriminator(context, reply, dis, dis_optimizer, gen, corpus)
