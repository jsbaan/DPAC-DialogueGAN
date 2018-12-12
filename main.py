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

import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

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

# from generator import Generator
from generator2 import Generator2

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda:0')
#     print("RUNNIG ON CUDA") #'
# else:
#     DEVICE = torch.device('cpu')  #'cuda:0'
#     print("RUNNING ON CPU")

DEVICE = torch.device('cpu')


VOCAB_SIZE = 8000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
DIS_TRAIN_EPOCHS = 50

GEN_EMBEDDING_DIM = 256
GEN_HIDDEN_DIM = 256
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

CAPACITY_RM = 100000
PRETRAIN = False
ACTOR_CHECKPOINT = "generator_checkpoint79.pth.tar"
GEN_MLE_LR = 1e-3
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
DISCRIMINATOR_LM = True     # one of the two (DISCRIMINATOR_LM or MC) must be False
MC = False
AC_WARMUP = 1000
DISCOUNT_FACTOR = 0.99

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

def train_generator_MLE(gen, optimizer, data, epochs):
    # Max Likelihood Pretraining for the generator
    corpus = data.dataset.corpus
    pad_id = corpus.token_to_id(corpus.PAD)

    loss_func = torch.nn.NLLLoss(ignore_index=pad_id)
    loss_func.to(DEVICE)

    start_epoch = 0
    # saved_data = try_get_state_dicts()
    # if saved_data is not None:
    #     start_epoch = saved_data['epoch']
    #     gen.load_state_dict(saved_data['state_dict'])
    #     optimizer.load_state_dict(saved_data['optimizer'])

    loss_per_epoch = []
    for epoch in range(start_epoch, epochs):
        print('epoch %d : ' % (epoch + 1))

        total_loss = 0
        losses = []
        for (iter, (context, reply)) in enumerate(train_data_loader):
            optimizer.zero_grad()
            context = context.permute(1,0)
            reply = reply.permute(1,0)
            output = gen.forward(context, reply)

            # Compute loss
            pred_dist = output[1:].view(-1, VOCAB_SIZE).to(DEVICE)
            tgt_tokens = reply[1:].contiguous().view(-1).to(DEVICE)

            loss = loss_func(pred_dist, tgt_tokens)

            # Backpropagate loss
            loss.backward()
            clip_grad_norm_(gen.parameters(), 10)
            optimizer.step()
            total_loss += loss.data.item()
            losses.append(loss)

            # Print updates
            if iter % 50 == 0 and iter != 0:
                print('[Epoch {} iter {}] loss: {}'.format(epoch,iter,total_loss/50))
                total_loss = 0
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': gen.state_dict(),
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

def train_generator_PG(context, reply, gen, gen_opt, dis):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for one batch.
    """
    # Forward pass
    fake_reply, word_probabilities = gen.sample(context, reply)
    # print("Generated reply")
    # print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
    # print("Real  reply")
    # print(corpus.ids_to_tokens([int(i) for i in reply[0]]))
    entropy = torch.mean(word_probabilities.log(), dim=1)
    perplexity = torch.mean(2**(-entropy)).item()

    # Compute word-level rewards
    rewards = dis.get_rewards(fake_reply, PAD)

    # Compute REINFORCE loss with the assumption that G = R_t
    pg_loss = gen.compute_reinforce_loss(rewards, word_probabilities)
    # Backward pass
    gen_opt.zero_grad()
    pg_loss.backward()
    gen_opt.step()
    return perplexity

def train_generator_PGAC(context, reply, gen, dis, memory, critic, AC_optimizer, \
        EOU,PAD):
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
    samples = torch.autograd.Variable(PAD*torch.ones(BATCH_SIZE,MAX_SEQ_LEN)).to(DEVICE)
    samples[:,0] = input
    active_ep_idx = torch.ones(BATCH_SIZE)
    EOU = torch.tensor(EOU).repeat(BATCH_SIZE)

    # Pass through decoder and sample action (word) from resulting vocab distribution
    for t in range(1, MAX_SEQ_LEN):
        output, hidden, attn_weights = gen.decoder(
                input, hidden, encoder_output)

        # Sample action (token) for entire batch from predicted vocab distribution
        # and set input for next forward pass
        action = torch.multinomial(torch.exp(output), 1).view(-1).data
        log_p = output.gather(1, action.unsqueeze(1)).view(-1).data
        input = torch.autograd.Variable(action)

        # Check which episodes (sampled sentences) have not encountered a EOU token
        done = (action == EOU).float()
        active_index = active_ep_idx.nonzero().squeeze(1)

        # Only put states of active episodes in replay memory
        old_state = samples.clone()
        reward = dis.get_reward(samples[active_index,:t], action[active_index])
        samples[:, t] = action
        done_index = done.nonzero()
        active_ep_idx[done_index] = 0

        for j,i in enumerate(active_index):
            memory.push((old_state[i,:], action[i], log_p[i], reward[j], samples[i,:], done[i]))

        if memory.__len__() > AC_WARMUP:
            # Retrieve batch from replay memory
            info = tuple(zip(*memory.sample(BATCH_SIZE)))
            state, action, log_p, reward, next_state, done = [torch.stack(i) for i in info]

            # Estimate state-action values for each state in batch using critic
            q_values = critic.forward(state.long())[np.arange(BATCH_SIZE), action]
            with torch.no_grad():
                mask = (done==False).float()
                q_values_target = mask.float()*(DISCOUNT_FACTOR * \
                    torch.max(critic.forward(next_state.long()), dim=1)[0].float()) \
                    + reward

            # TODO add rho importance sampling
            # Compute combined actor critic loss and backprop
            actor_loss = -torch.mean(q_values)
            critic_loss = F.smooth_l1_loss(q_values, q_values_target)
            loss = actor_loss + critic_loss
            AC_optimizer.zero_grad()
            loss.backward()
            AC_optimizer.step()
    return loss


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

def calc_mean(rewards):

    batch_size, length = rewards.shape

    total = 0
    for i in range(batch_size):
        reward = rewards[i]
        idx = (reward == 0).nonzero()
        if len(idx) > 0:
            idx = idx[0].item()
        else:
            idx = length
        total += torch.mean(reward[0:idx])
    return total/batch_size

def train_discriminator(gen, dis, dis_opt):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    for (iter, (context, real_reply)) in enumerate(train_data_loader):

        dis_opt.zero_grad()

        with torch.no_grad():
            fake_reply, _= gen.sample(context, real_reply)
        fake_reply = fill_with_padding(fake_reply, EOU, PAD)

        if DISCRIMINATOR_LM:
   
            real_r = dis.get_rewards(real_reply, PAD)
            fake_r = dis.get_rewards(fake_reply, PAD)

            real_rewards = calc_mean(real_r)
            fake_rewards = calc_mean(fake_r)

            loss = -(real_rewards - fake_rewards)

        else:
            fake_targets = torch.zeros(BATCH_SIZE)
            real_targets = torch.ones(BATCH_SIZE)

            dis_opt.zero_grad()
            out_fake = dis.batchClassify(context, fake_reply.long())
            out_real = dis.batchClassify(context, real_reply.long())

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
    

def pre_train_discriminator(dis, dis_opt, gen, corpus, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

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

            with torch.no_grad():
                fake_reply, _ = gen.sample(context, real_reply)
            fake_reply = fill_with_padding(fake_reply, EOU, PAD)
     
            real_r = dis.get_rewards(real_reply, PAD)
            fake_r = dis.get_rewards(fake_reply, PAD)

            real_rewards = calc_mean(real_r)
            fake_rewards = calc_mean(fake_r)

            loss = -(real_rewards - fake_rewards)

            loss.backward()
            dis_opt.step()
            total_loss += loss.data.item()
            losses.append(loss)

            if iter % 20 == 0:
                print("loss ", loss.item())

            # print updates
            # if iter % 50 == 0 and iter != 0:
            if iter % 50 == 0:
                print('[Epoch {} iter {}] loss: {}'.format(epoch,iter,total_loss/50))
                total_loss = 0
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': dis.state_dict(),
                    'optimizer' : dis_opt.state_dict(),
                    'loss'      : losses,
                },'discriminator_checkpoint{}.pth.tar'.format(epoch))

                try:
                    print("Fake generated reply")
                    print(corpus.ids_to_tokens([int(i) for i in fake_reply[0]]))
                    print("Real  reply")
                    print(corpus.ids_to_tokens([int(i) for i in real_reply[0]]))

                    print("fake reward", calc_mean(fake_r[0].unsqueeze(0)).item())
                    print("real reward", calc_mean(real_r[0].unsqueeze(0)).item())

                    print("mean fake reward ", torch.mean(fake_rewards).item())
                    print("mean real reward ", torch.mean(real_rewards).item())
                except:
                    print("Unable to print")


        loss_per_epoch.append(total_loss)
    torch.save(loss_per_epoch, "discriminator_final_loss.pth.tar")

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
    SOS = train_data_loader.dataset.corpus.token_to_id(DPCorpus.SOS)
    EOU = train_data_loader.dataset.corpus.token_to_id(DPCorpus.EOU)
    PAD = train_data_loader.dataset.corpus.token_to_id(DPCorpus.PAD)

    # Initalize Networks and optimizers
    gen = Generator2(SOS,EOU, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN)
    genMLE_optimizer = optim.Adam(gen.parameters(), lr = GEN_MLE_LR)
    dis = discriminator_LM.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE).to(DEVICE)
    dis_optimizer = optim.Adagrad(dis.parameters()) ## ADAGRAD ??

    # # Pretrain generator and discriminator
    # if PRETRAIN:
    #     print('Starting Generator MLE Training...')
    #     train_generator_MLE(gen, genMLE_optimizer, train_data_loader, MLE_TRAIN_EPOCHS)

    #     print('\nStarting Discriminator Training...')
    #     for epoch in range(ADV_TRAIN_EPOCHS):
    #         for (batch, (context, reply)) in enumerate(train_data_loader):
    #             train_discriminator(context, reply, dis, dis_optimizer, gen, corpus)

    # # ADVERSARIAL TRAINING
    # # Initialize actor as pre-trained generator
    # actor = Generator2(SOS,EOU, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN)
    # actor.load_state_dict(torch.load(ACTOR_CHECKPOINT, map_location='cpu')['state_dict'])
    # PG_optimizer = optim.Adam(actor.parameters(),ACTOR_LR)

    # # Define critic and dual optimizer
    # critic = critic.Critic(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)
    # AC_optimizer = optim.Adam([
    #     {'params': gen.parameters(), 'lr': ACTOR_LR},
    #     {'params': critic.parameters(), 'lr': CRITIC_LR}
    # ])
    # memory = replay_memory.ReplayMemory(CAPACITY_RM)

    # print('\nStarting Adversarial Training...')
    # for epoch in range(ADV_TRAIN_EPOCHS):
    #     print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    #     sys.stdout.flush()
    #     for (batch, (context, reply)) in enumerate(train_data_loader):
    #         # TRAIN GENERATOR
    #         print('\nAdversarial Training Generator: ')
    #         perplexity = train_generator_PG(context, reply,\
    #         actor, PG_optimizer,dis)
    #         # perplexity = train_generator_PGAC(context, reply,\
    #         #     gen, dis, memory, critic, AC_optimizer,EOU,PAD)
    #         if batch % 10 == 0:
    #             print("After " + str(batch) + " batches, the perplexity is: " + str(perplexity))

    #         # TRAIN DISCRIMINATOR
    #         print('\nAdversarial Training Discriminator : ')
    #         train_discriminator(actor, dis, dis_optimizer)


    # PRETRAINING DISCRIMINATOR

    # Load pretrained generator
    saved_gen = torch.load('generator_checkpoint79.pth.tar')
    gen.load_state_dict(saved_gen['state_dict'])


    # start pretraining
    print('\nStarting Discriminator Training...')
    pre_train_discriminator(dis, dis_optimizer, gen, corpus, DIS_TRAIN_EPOCHS)