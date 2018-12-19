from seq2seq.TopKDecoder import TopKDecoder
from helpers import *
from generator import Generator
# from Evaluator import Evaluator
# from __future__ import print_function

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
import numpy as np
from evaluation.Evaluator import Evaluator
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')  #'
else:
    DEVICE = torch.device('cpu')  #'cuda:0'

VOCAB_SIZE = 8000
BATCH_SIZE = 64
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
BATCH_SIZE = 64
GEN_EMBEDDING_DIM = 256
GEN_HIDDEN_DIM = 256

def load_data(path='dataset.pickle'):
    """
    Load data set
    """
    if not os.path.isfile(path):
        # print("Saving the data set")
        corpus = DPCorpus(vocabulary_limit=VOCAB_SIZE)
        train_dataset = corpus.get_train_dataset(min_reply_length=MIN_SEQ_LEN,\
            max_reply_length=MAX_SEQ_LEN)

        with open(path, 'wb') as handle:
            pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        train_data_loader = DPDataLoader(train_dataset, batch_size=BATCH_SIZE)

    else:
        # print("Loading the data set")
        with open(path, 'rb') as handle:
            train_dataset = pickle.load(handle)
        train_data_loader = DPDataLoader(train_dataset, batch_size=BATCH_SIZE)
    return train_data_loader

if __name__ == '__main__':
    # evaluator = Evaluator(vocab_size=VOCAB_SIZE, min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE)
    train_data_loader = load_data()
    corpus = train_data_loader.dataset.corpus
    SOS = train_data_loader.dataset.corpus.token_to_id(DPCorpus.SOS)
    EOU = train_data_loader.dataset.corpus.token_to_id(DPCorpus.EOU)
    PAD = train_data_loader.dataset.corpus.token_to_id(DPCorpus.PAD)
    corpus = train_data_loader.dataset.corpus
    dataset = corpus.get_validation_dataset(min_reply_length=MIN_SEQ_LEN, max_reply_length=MAX_SEQ_LEN)
    dataset = DPDataLoader(dataset, batch_size=BATCH_SIZE)

    gen_pretrain = Generator(SOS, EOU, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, teacher_forcing_ratio=0)
    gen_seq = Generator(SOS, EOU, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, teacher_forcing_ratio=0)
    model_path_pretrain = 'generator_checkpoint79.pth.tar'
    model_path_seq = 'adversial_checkpoint3_seq_gan.pth.tar'
    data_pretrain = torch.load(model_path_pretrain, map_location='cpu')
    data_seq = torch.load(model_path_seq, map_location='cpu')

    gen_pretrain.load_state_dict(data_pretrain['state_dict'])
    gen_pretrain.decoder = TopKDecoder(gen_pretrain.decoder, 5)
    gen_pretrain.to(DEVICE)

    gen_seq.load_state_dict(data_seq['actor'])
    gen_seq.decoder = TopKDecoder(gen_seq.decoder, 5)
    gen_seq.to(DEVICE)

    iterator = iter(dataset)
    context, reply = iterator.next()
    prediction_pretrain, meta_data_pretrain = gen_pretrain(context.t(), reply.t(), hack=True)
    prediction_seq, meta_data_seq = gen_seq(context.t(), reply.t(), hack=True)

    beam_pretrain = torch.stack(meta_data_pretrain['sequence']).squeeze(2).t()
    beam_seq = torch.stack(meta_data_seq['sequence']).squeeze(2).t()


    for i, batch in enumerate(context):
        print("CONTEXT:")
        print(', '.join(corpus.ids_to_tokens([int(word) for word in batch])))
        print("REAL REPLY:")
        print(', '.join(corpus.ids_to_tokens([int(word) for word in reply[i]])))
        print("PRETRAIN REPLY")
        print(', '.join(corpus.ids_to_tokens([int(word.item()) for word in beam_pretrain[i]])))
        print("SEQ REPLY")
        print(', '.join(corpus.ids_to_tokens([int(word.item()) for word in beam_seq[i]])))
        print("-" * 50)

        # print(', '.join(corpus.ids_to_tokens([int(word.item()) for word in prediction_pretrain[i]])))

    # print(prediction)




   
