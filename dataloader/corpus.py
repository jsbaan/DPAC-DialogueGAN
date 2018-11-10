from collections import Counter
from daily_dialog_parser import DailyDialogParser
from dp_dataset import DPDataset

class Corpus(object):
    PAD = '<pad>' # Padding token
    UNK = '<unk>' # Unknown token (Out of vocabulary)

    def __init__(self, path='daily_dialog/', parser=DailyDialogParser(), vocabulary_limit=None):
        self.train_corpus = parser.process_file(path + 'train.txt')
        self.validation_corpus = parser.process_file(path + 'validation.txt')
        self.test_corpus = parser.process_file(path + 'test.txt')

        print('Building vocabulary')
        self.build_vocab(vocabulary_limit)

        if vocabulary_limit is not None:
            print('Replacing out of vocabulary from train corpus by unk token.')
            self.limit_corpus_to_vocabulary(self.train_corpus)
            print('Replacing out of vocabulary from validation corpus by unk token.')
            self.limit_corpus_to_vocabulary(self.validation_corpus)
            print('Replacing out of vocabulary from test corpus by unk token.')
            self.limit_corpus_to_vocabulary(self.test_corpus)

    def build_vocab(self, vocabulary_limit):
        special_tokens = [self.PAD, self.UNK]
        all_words = self.flatten_corpus(self.train_corpus)

        vocabulary_counter = Counter(all_words)
        if vocabulary_limit is not None:
            vocabulary_counter = vocabulary_counter.most_common(vocabulary_limit - len(special_tokens))
        else:
            vocabulary_counter = vocabulary_counter.most_common()

        self.vocabulary = special_tokens + [token for token, _ in vocabulary_counter]
        self.token_ids = {token: index for index, token in enumerate(self.vocabulary)}

    def flatten_corpus(self, corpus):
        all_words = []
        for dialog in corpus:
            for utterance in dialog:
                all_words.extend(utterance)
        return all_words

    def limit_corpus_to_vocabulary(self, corpus):
        for d_i, dialog in enumerate(corpus):
            for u_i, utterance in enumerate(dialog):
                for t_i, token in enumerate(utterance):
                    if token not in self.vocabulary:
                        corpus[d_i][u_i][t_i] = self.UNK

    def utterance_to_ids(self, utterance):
        utterance_ids = []

        for token in utterance:
            utterance_ids.append(self.token_ids.get(token, self.token_ids[self.UNK]))

        return utterance_ids

    def corpus_to_ids(self, data):
        data_ids = []

        for dialog in data:
            dialog_ids = []

            for utterance in dialog:
                dialog_ids.append(self.utterance_to_ids(utterance))
            data_ids.append(dialog_ids)

        return data_ids

    def get_train_dataset(self, context_size=3):
        return self.get_dataset(self.train_corpus, context_size)

    def get_validation_dataset(self, context_size=3):
        return self.get_dataset(self.validation_corpus, context_size)

    def get_test_dataset(self, context_size=3):
        return self.get_dataset(self.test_corpus, context_size)

    def get_dataset(self, corpus, context_size):
        corpus_ids = self.corpus_to_ids(corpus)
        return DPDataset(corpus_ids, context_size)