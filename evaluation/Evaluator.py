from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
from torchnlp.metrics import get_moses_multi_bleu
import pickle
import os
from nlgeval import NLGEval
from embedding_metrics import *
import torch

class Evaluator:
    def __init__(self, data_loader_path='../dataloader/daily_dialog/', log=True, vocab_size = 8000, min_seq_len=5, max_seq_len=20, batch_size=128):
        self.log = log
        self.vocab_size = vocab_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.load_data_loader(data_loader_path + 'test_loader' + '_' + str(batch_size) + '.pickle')

        self.corpus = self.data_loader.dataset.corpus
        self.sos_id = self.corpus.token_to_id(self.corpus.SOS)
        self.eou_id = self.corpus.token_to_id(self.corpus.EOU)

    def load_data_loader(self, path):
        if not os.path.isfile(path):
            corpus = DPCorpus(vocabulary_limit=self.vocab_size)
            dataset = corpus.get_test_dataset(min_reply_length=self.min_seq_len, max_reply_length=self.max_seq_len)
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

        eval = NLGEval()
        return eval.compute_individual_metrics(ref=real_replies, hyp=generated_replies)

    def get_replies(self, model):
        real_replies = []
        generated_replies = []

        for (iter, (context, reply)) in enumerate(self.data_loader):
            if self.log:
                print(str(iter + 1) + '/' + str(len(self.data_loader)))

            context = context.permute(1, 0)
            reply = reply.permute(1, 0)
            output = model(context, reply)

            for i in range(context.size(1)):
                context_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in context[:, i]]))
                real_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in reply[:, i]]))

                output_i = [int(i) for i in output.argmax(2)[:, i].tolist()]
                try:
                    eou_i = output.index(self.eou_id)
                    output_i = output[:eou_i + 1]
                except:
                    pass
                generated_i = ' '.join(self.corpus.ids_to_tokens([int(i) for i in output_i]))

                real_replies.append(real_i)
                generated_replies.append(generated_i)

        return real_replies, generated_replies

    def get_word2vec(self, embedding_model, replies):
        word2vec = {}

        for reply in replies:
            tokens = reply.split()

            for token in tokens:
                if token not in word2vec:
                    id = self.corpus.token_to_id(token)
                    id_tensor = torch.tensor(id, dtype=torch.long, requires_grad=False)
                    embedding = embedding_model(id_tensor)
                    word2vec[token] = embedding

        return word2vec