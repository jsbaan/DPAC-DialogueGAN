from __future__ import print_function

import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from seq2seq.TopKDecoder import TopKDecoder

import discriminator
import discriminator_LM
from helpers import *
from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os
from torchnlp.metrics import *

from generator import Generator
from generator2 import Generator2

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')  #'
else:
    DEVICE = torch.device('cpu')  #'cuda:0'
VOCAB_SIZE = 8000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50

GEN_EMBEDDING_DIM = 256
GEN_HIDDEN_DIM = 256
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64
DISCRIMINATOR_LM = False     # one of the two (DISCRIMINATOR_LM or MC) must be False
MC = True

def get_data():

    if not os.path.isfile('dataloader/daily_dialog/test_loader.pickle'):
        corpus = DPCorpus(vocabulary_limit=VOCAB_SIZE)
        test_dataset = corpus.get_test_dataset(min_reply_length=MIN_SEQ_LEN, max_reply_length=MAX_SEQ_LEN)
        test_data_loader = DPDataLoader(test_dataset, batch_size=BATCH_SIZE)

        with open('dataloader/daily_dialog/test_loader.pickle', 'wb') as f:
            pickle.dump(test_data_loader, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('dataloader/daily_dialog/test_loader.pickle', 'rb') as f:
            test_data_loader= pickle.load(f)
        corpus = test_data_loader.dataset.corpus

    return corpus, test_data_loader

if __name__ == '__main__':
    corpus, test_data_loader = get_data()

    sos_id = corpus.token_to_id(corpus.SOS)
    eou_id = corpus.token_to_id(corpus.EOU)
    gen = Generator2(sos_id, eou_id, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, teacher_forcing_ratio=0)

    data = torch.load('./generator_checkpoint59.pth.tar', map_location='cpu')
    gen.load_state_dict(data['state_dict'])

    gen.decoder = TopKDecoder(gen.decoder, 20)
    gen.to(DEVICE)

    real_replies = []
    generated_replies = []

    for (iter, (context, reply)) in enumerate(test_data_loader):
        print(str(iter+1) + '/' + str(len(test_data_loader)))

        context = context.permute(1,0).to(DEVICE)
        reply = reply.permute(1,0).to(DEVICE)
        output = gen.forward(context, reply)

        for i in range(context.size(1)):
            context_i = ' '.join(corpus.ids_to_tokens([int(i) for i in context[:,i]]))
            real_i = ' '.join(corpus.ids_to_tokens([int(i) for i in reply[:,i]]))

            output_i = [int(i) for i in output.argmax(2)[:, i].tolist()]
            try:
                eou_i = output.index(eou_id)
                output_i = output[:eou_i + 1]
            except:
                pass
            generated_i = ' '.join(corpus.ids_to_tokens([int(i) for i in output_i]))

            real_replies.append(real_i)
            generated_replies.append(generated_i)

            if i == 0:
                print('Context')
                print(context_i)
                print('Real reply')
                print(real_i)
                print('Generated reply')
                print(generated_i)
                print()

    bleu = get_moses_multi_bleu(generated_replies, real_replies)
    print('BLUE')
    print(bleu)
