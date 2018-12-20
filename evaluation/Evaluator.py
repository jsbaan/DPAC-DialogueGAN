from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
import pickle
import os

try:
    from nlgeval import NLGEval
except:
    pass

from evaluation.embedding_metrics import *
from evaluation.distinct_metrics import *
import torch
# from torchnlp.metrics import *

import word2vec

class Evaluator:
    def __init__(self, data_loader_path=None, verbose=False, vocab_size = 8000, min_seq_len=5, max_seq_len=20, batch_size=128, device="cpu"):
        self.verbose = verbose
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

    def evaluate(self, model, distinct=False, nlg=False, embedding=False):
        real_replies, generated_replies = self.get_replies(model)

        all_results = {}
        if distinct:
            result = self.evaluate_distinct(model, real_replies=real_replies, generated_replies=generated_replies)
            all_results = {**all_results, **result}
        if nlg:
            result = self.evaluate_nlg(model, real_replies=real_replies, generated_replies=generated_replies)
            all_results = {**all_results, **result}
        if embedding:
            result = self.evaluate_embeddings(model, real_replies=real_replies, generated_replies=generated_replies)
            all_results = {**all_results, **result}

        return all_results

    def evaluate_embeddings(self, model, real_replies=None, generated_replies=None, real_path='real.txt', generated_path='generated.txt'):
        if real_replies is None or generated_replies is None:
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

    def evaluate_distinct(self, model, real_replies=None, generated_replies=None):
        if real_replies is None or generated_replies is None:
            real_replies, generated_replies = self.get_replies(model)

        dist_1 = distinct_ngrams(1, generated_replies)
        dist_2 = distinct_ngrams(2, generated_replies)
        dist_3 = distinct_ngrams(3, generated_replies)
        dist_s = distinct_sentences(generated_replies)
        tokens = token_count(generated_replies)

        result = {
            'dist_1' : dist_1,
            'dist_2' : dist_2,
            'dist_3' : dist_3,
            'dist_s' : dist_s,
            'tokens' : tokens
        }

        return result

    def evaluate_nlg(self, model, real_replies=None, generated_replies=None):
        if real_replies is None or generated_replies is None:
            real_replies, generated_replies = self.get_replies(model)

        eval = NLGEval()
        return eval.compute_metrics([real_replies], generated_replies)

    def get_replies(self, model):
        real_replies = []
        generated_replies = []

        for (iter, (context, reply)) in enumerate(self.data_loader):
            if self.verbose:
                print(str(iter + 1) + '/' + str(len(self.data_loader)))
            context = context.permute(1, 0).to(self.device)
            reply = reply.permute(1, 0).to(self.device)
            _, meta_data = model(context, reply, hack=True)

            output = torch.stack(meta_data['sequence']).squeeze(2).t().tolist()
            for i in range(context.size(1)):
                context_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in context[:, i]]))
                real_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in reply[:, i]]))# if i not in self.tokens_to_remove]))

                output_i = output[i]
                try:
                    eou_i = output_i.index(self.eou_id)
                    output_i = output_i[:eou_i + 1]
                except:
                    pass

                generated_i = ' '.join([self.corpus.SOS] + self.corpus.ids_to_tokens([int(i) for i in output_i]))# if i not in self.tokens_to_remove]))

                if self.verbose and i == 0:
                    print(context_i)
                    print(real_i)
                    print(generated_i)
                    print()

                real_replies.append(real_i)
                generated_replies.append(generated_i)
                # break

            # break
        return real_replies, generated_replies

    def get_word2vec(self, embedding_model, replies):
        path = os.path.dirname(os.path.realpath(__file__))
        w2v = word2vec.load(path + '/word2vec.bin')
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