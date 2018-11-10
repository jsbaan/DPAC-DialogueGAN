from collections import Counter
from daily_dialog_parser import DailyDialogParser

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




    def sentence_to_ids(self, sentence):
        sentence_ids = []

        for token in sentence:
            sentence_ids.append(self.reversed_vocab.get(token, default=self.unk_id))

        return sentence_ids

    def corpus_to_ids(self, data):
        data_ids = []

        for dialog in data:
            dialog_ids = []

            for sentence in dialog:
                dialog_ids.append(self.sentence_to_ids(sentence))
            data_ids.append(dialog_ids)

        return results

    def get_corpus_ids(self):
        train_ids = self.corpus_to_ids(self.train_corpus)
        validation_ids = self.corpus_to_ids(self.validation_corpus)
        test_ids = self.corpus_to_ids(self.test_corpus)

        return train_ids, validation_ids, test_ids