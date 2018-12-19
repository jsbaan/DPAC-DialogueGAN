from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os

# try:
#     from nlgeval import NLGEval
# except:
#     pass

from evaluation.embedding_metrics import *
import torch
# from torchnlp.metrics import *

import word2vec

class Evaluator:
    def __init__(self, data_loader_path=None, log=True, vocab_size = 8000, min_seq_len=5, max_seq_len=20, batch_size=128, device="cpu"):
        self.log = log
        self.vocab_size = vocab_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        if data_loader_path == None:
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
            data_loader_path = parent_dir + '/dataloader/daily_dialog/'
        self.load_data_loader(data_loader_path + 'validation_loader' + '_' + str(batch_size) + '.pickle')

        self.corpus = self.data_loader.dataset.corpus
        self.sos_id = self.corpus.token_to_id(self.corpus.SOS)
        self.eos_id = self.corpus.token_to_id(self.corpus.EOS)
        self.eou_id = self.corpus.token_to_id(self.corpus.EOU)
        self.tokens_to_remove = [self.sos_id, self.eos_id, self.eou_id]
        self.device = device

    def load_data_loader(self, path):
        if not os.path.isfile(path):
            corpus = DPCorpus(vocabulary_limit=self.vocab_size)
            dataset = corpus.get_validation_dataset(min_reply_length=self.min_seq_len, max_reply_length=self.max_seq_len)
            self.data_loader = DPDataLoader(dataset, batch_size=self.batch_size)

            with open(path, 'wb') as f:
                pickle.dump(self.data_loader, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'rb') as f:
                self.data_loader= pickle.load(f)

    def evaluate_embeddings(self, model, real_path='real.txt', generated_path='generated.txt'):
        real_replies, generated_replies = self.get_replies(model)

        with open(real_path, 'w') as file:
            for reply in real_replies:
                file.write("%s\n" % reply)

        with open(generated_path, 'w') as file:
            for reply in generated_replies:
                file.write("%s\n" % reply)

        embedding_model = model.encoder.embedding
        word2vec = self.get_word2vec(embedding_model, real_replies+generated_replies)

        result = {
            'greedy_match' : greedy_match(real_path, generated_path, word2vec),
            'extrema_score' : extrema_score(real_path, generated_path, word2vec),
            'average' : average(real_path, generated_path, word2vec)
        }

        return result

    def evaluate_nlg(self, model):
        real_replies, generated_replies = self.get_replies(model)

        # real_replies = [[r] for r in real_replies]

        eval = NLGEval()
        return eval.compute_metrics(real_replies, generated_replies)

    def get_replies(self, model):
        real_replies = []
        generated_replies = []

        for (iter, (context, reply)) in enumerate(self.data_loader):
            # if self.log:
            # print(str(iter + 1) + '/' + str(len(self.data_loader)))
            context = context.permute(1, 0).to(self.device)
            reply = reply.permute(1, 0).to(self.device)
            output = model(context, reply)

            for i in range(context.size(1)):
                context_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in context[:, i]]))
                real_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in reply[:, i]]))# if i not in self.tokens_to_remove]))

                output_i = [int(i) for i in output.argmax(2)[:, i].tolist()]
                try:
                    eou_i = output_i.index(self.eou_id)
                    output_i = output_i[:eou_i + 1]
                except:
                    pass

                generated_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in output_i]))# if i not in self.tokens_to_remove]))

                # if i == 0:
                #     print(context_i)
                #     print(real_i)
                #     print(generated_i)
                #     print()

                real_replies.append(real_i)
                generated_replies.append(generated_i)
                # break

            # break
        return real_replies, generated_replies

    def get_word2vec(self, embedding_model, replies):
        path = os.path.dirname(os.path.realpath(__file__))
        w2v = word2vec.load(path + '/word2vec.bin')

        #torchwordemb.load_word2vec_bin('GoogleNews-vectors-negative300.bin')
        # word2vec = {}
        # embedding_model = embedding_model.to(self.device)
        # for reply in replies:
        #     tokens = reply.split()
        #
        #     for token in tokens:
        #         if token not in word2vec:
        #             # id = self.corpus.token_to_id(token)
        #             # id_tensor = torch.tensor(id, dtype=torch.long, requires_grad=False)
        #             # embedding = embedding_model(id_tensor.to(self.device))
        #             # word2vec[token] = embedding
        #             word2vec[token] = vec[vocab[token]]
        #
        # word2vec['<unk>'] = torch.zeros(300)

        return WordVectorsWrapper(w2v)

class WordVectorsWrapper:
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        self.layer1_size = 100

    def __getitem__(self, item):
        try:
            result = torch.from_numpy(self.word_vectors[item])
            return result
        except:
            return torch.rand(100)

        # if item in self.word_vectors:
        #     return torch.from_numpy(self.word_vectors[item])
        # else:
        #     return torch.rand(256)