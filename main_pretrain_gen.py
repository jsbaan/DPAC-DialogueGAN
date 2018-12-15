from __future__ import print_function

import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import discriminator_LM
import critic

from helpers import *
from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os
import time
import replay_memory
from evaluation.Evaluator import Evaluator


from generator import Generator

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print("RUNNIG ON CUDA") #'
else:
    DEVICE = torch.device('cpu')  #'cuda:0'
    print("RUNNING ON CPU")

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
PRETRAIN_GENERATOR = True
PRETRAIN_DISCRIMINATOR = False
POLICY_GRADIENT = False
ACTOR_CHECKPOINT = "generator_checkpoint79.pth.tar"
DISCRIMINATOR_CHECKPOINT = "discriminator_epoch6_iter.txt"
GEN_MLE_LR = 1e-3
DISCRIMINATOR_MLE_LR = 3e-3
ACTOR_LR = 3e-3
CRITIC_LR = 3e-3
DISCRIMINATOR_LR = 3e-3
AC = False
AC_WARMUP = 1000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE_TESTING = 256
# Number of gen

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

def train_generator_PGAC(context, reply, gen, dis, memory, critic, AC_optimizer, EOU,PAD):
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
            split = torch.split(sent, idx+1)[0].to(DEVICE)
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

def train_discriminator(context,real_reply,gen, dis, dis_opt):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    dis_opt.zero_grad()

    with torch.no_grad():
        fake_reply, _= gen.sample(context, real_reply)
    fake_reply = fill_with_padding(fake_reply, EOU, PAD)

    real_r = dis.get_rewards(real_reply, PAD)
    fake_r = dis.get_rewards(fake_reply, PAD)

    real_rewards = calc_mean(real_r)
    fake_rewards = calc_mean(fake_r)

    loss = -(real_rewards - fake_rewards)

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
            context = context.to(DEVICE)
            real_reply = real_reply.to(DEVICE)

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
        train_data_loader = DPDataLoader(train_dataset, batch_size=BATCH_SIZE)
        train_MLE_data_loader = DPDataLoader(train_dataset, batch_size=BATCH_SIZE)

        with open(path, 'wb') as handle:
            pickle.dump(train_data_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            corpus = train_data_loader.dataset.corpus
    else:
        print("Loading the data set")
        with open(path, 'rb') as handle:
            train_data_loader= pickle.load(handle)
        train_MLE_data_loader = DPDataLoader(train_data_loader.dataset, batch_size=BATCH_SIZE)
        corpus = train_data_loader.dataset.corpus
    return corpus,train_data_loader, train_MLE_data_loader

def save_models(actor, discriminator, epoch, PG_optimizer, actorMLE_optimizer, dis_optimizer):
    torch.save({
                        'epoch': epoch+1,
                        'actor': actor.state_dict(),
                        'act_optimizer' : PG_optimizer.state_dict(),
                        'act_MLE_optimizer' : actorMLE_optimizer.state_dict(),
                        'dis_optimizer' : dis_optimizer.state_dict(),
                        'discriminator': discriminator.state_dict()
                    },'adversial_checkpoint{}.pth.tar'.format(epoch))
    print("Models and Optimizers saved")


def perform_evaluation(evaluator, actor):
    actor = actor.eval()
    result = evaluator.evaluate_embeddings(actor)
    print("Evaluation")
    print("Greedy Match: ", result['greedy_match'][0])
    print("Extrema Score: ", result['extrema_score'][0])
    print("Average (Cosine similarity): ", result['average'][0])
    actor = actor.train()

if __name__ == '__main__':
    '''
    Main training loop. Pre-trains the generator and discriminator using MLE
    and then uses PG to alternately train them.
    '''
    # Load data set
    corpus, train_data_loader, MLE_data_loader = load_data()
    SOS = train_data_loader.dataset.corpus.token_to_id(DPCorpus.SOS)
    EOU = train_data_loader.dataset.corpus.token_to_id(DPCorpus.EOU)
    PAD = train_data_loader.dataset.corpus.token_to_id(DPCorpus.PAD)

    # Pretrain generator and discriminator
    if PRETRAIN_GENERATOR:
        print('Starting Generator MLE Training...')
        gen = Generator(SOS,EOU,VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN).to(DEVICE)
        genMLE_optimizer = optim.Adam(gen.parameters(), lr = GEN_MLE_LR)
        gen.train_generator_MLE(genMLE_optimizer, train_data_loader, MLE_TRAIN_EPOCHS)

    if PRETRAIN_DISCRIMINATOR:
        print('\nStarting Discriminator MLE Training...')
        # Initialize discriminator
        dis = discriminator_LM.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE).to(DEVICE)
        dis_optimizer = optim.Adam(dis.parameters(),lr = DISCRIMINATOR_MLE_LR)

        # Load pretrained generator
        gen = Generator(SOS,EOU,VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN).to(DEVICE)
        saved_gen = torch.load(ACTOR_CHECKPOINT)
        gen.load_state_dict(saved_gen['state_dict'])
        pre_train_discriminator(dis, dis_optimizer, gen, corpus, DIS_TRAIN_EPOCHS)
    if POLICY_GRADIENT:
        ## ADVERSARIAL TRAINING
        # Initialize actor and discriminator using pre-trained state-dict
        actor = Generator(SOS,EOU, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM,\
            MAX_SEQ_LEN).to(DEVICE)
        actor.load_state_dict(torch.load(ACTOR_CHECKPOINT,map_location=DEVICE)['state_dict'])
        actorMLE_optimizer = optim.Adagrad(actor.parameters(),lr=GEN_MLE_LR)
        discriminator = discriminator_LM.Discriminator(DIS_EMBEDDING_DIM, \
        DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE).to(DEVICE)
        if DISCRIMINATOR_CHECKPOINT:
            discriminator.load_state_dict(torch.load(DISCRIMINATOR_CHECKPOINT,map_location=DEVICE)['state_dict'])
        dis_optimizer = optim.Adagrad(discriminator.parameters(),lr=DISCRIMINATOR_LR)
        evaluator = Evaluator(vocab_size=VOCAB_SIZE, min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE_TESTING, device=DEVICE)

        # Define critic and dual optimizer
        if AC:
            critic = critic.Critic(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, device=DEVICE)
            AC_optimizer = optim.Adagrad([
                {'params': actor.parameters(), 'lr': ACTOR_LR},
                {'params': critic.parameters(), 'lr': CRITIC_LR}
            ])
            memory = replay_memory.ReplayMemory(CAPACITY_RM)
        # Use optimizer for baseline DP-GAN
        else:
            PG_optimizer = optim.Adagrad(actor.parameters(),ACTOR_LR)
        # Evaluation
        print("Pretrained evaluation")
        for epoch in range(ADV_TRAIN_EPOCHS):
            perform_evaluation(evaluator, actor)
            if epoch % 3 == 0 and epoch > 0:
                save_models(actor, discriminator, epoch, PG_optimizer, actorMLE_optimizer, dis_optimizer)

            dataiter = iter(MLE_data_loader)
            print('\n--------\nEPOCH %d\n--------' % (epoch+1))



            sys.stdout.flush()
            for (batch, (context, reply)) in enumerate(train_data_loader):
                context = context.to(DEVICE)
                reply = reply.to(DEVICE)
                # TRAIN GENERATOR (ACTOR)
                # Policy gradient step
                if AC:
                    perplexity = train_generator_PGAC(context, reply,\
                        actor, discriminator, memory, critic, AC_optimizer,EOU,PAD)
                # Or actor critic step
                else:
                    perplexity = train_generator_PG(context, reply,\
                    actor, PG_optimizer,discriminator)

                ## MLE step
                context_MLE, reply_MLE = dataiter.next()
                actor.train_generator_MLE_batch(context_MLE.to(DEVICE), reply_MLE.to(DEVICE), actorMLE_optimizer, PAD)

                # TRAIN DISCRIMINATOR
                train_discriminator(context,reply, actor, discriminator, dis_optimizer)
    print("DO NOT FORGET TO SAVE YOUR DATA IF YOU ARE RUNNING IN COLLAB")
