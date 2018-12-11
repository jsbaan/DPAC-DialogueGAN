from collections import Counter
from .daily_dialog_parser import DailyDialogParser
from .dp_dataset import DPDataset
from .dp_collator import DPCollator
import os

class DPCorpus(object):
    SOS = '<s>' # Start of sentence token
    EOS = '</s>' # End of sentence token
    EOU = '</u>' # End of utterance token
    PAD = '<pad>' # Padding token
    UNK = '<unk>' # Unknown token (Out of vocabulary)

    def __init__(self, dialog_parser=None, vocabulary_limit=None):
        if dialog_parser is None:
            path = os.path.dirname(os.path.realpath(__file__)) + '/daily_dialog/'
            dialog_parser = DailyDialogParser(path, self.SOS, self.EOS, self.EOU)

        self.train_dialogs, self.validation_dialogs, self.test_dialogs = dialog_parser.get_dialogs()

        print('Building vocabulary')
        self.build_vocab(vocabulary_limit)

        if vocabulary_limit is not None:
            print('Replacing out of vocabulary from train dialogs by unk token.')
            self.limit_dialogs_to_vocabulary(self.train_dialogs)
            print('Replacing out of vocabulary from validation dialogs by unk token.')
            self.limit_dialogs_to_vocabulary(self.validation_dialogs)
            print('Replacing out of vocabulary from test dialogs by unk token.')
            self.limit_dialogs_to_vocabulary(self.test_dialogs)

    def build_vocab(self, vocabulary_limit):
        special_tokens = [self.PAD, self.UNK]
        all_words = self.flatten_dialogs(self.train_dialogs)

        vocabulary_counter = Counter(all_words)
        if vocabulary_limit is not None:
            vocabulary_counter = vocabulary_counter.most_common(vocabulary_limit - len(special_tokens))
        else:
            vocabulary_counter = vocabulary_counter.most_common()

        self.vocabulary = special_tokens + [token for token, _ in vocabulary_counter]
        self.token_ids = {token: index for index, token in enumerate(self.vocabulary)}


    def flatten_dialogs(self, dialogs):
        all_words = []
        for dialog in dialogs:
            for utterance in dialog:
                all_words.extend(utterance)
        return all_words

    def limit_dialogs_to_vocabulary(self, dialogs):
        for d_i, dialog in enumerate(dialogs):
            for u_i, utterance in enumerate(dialog):
                for t_i, token in enumerate(utterance):
                    if token not in self.vocabulary:
                        dialogs[d_i][u_i][t_i] = self.UNK

    def utterance_to_ids(self, utterance):
        utterance_ids = []

        for token in utterance:
            utterance_ids.append(self.token_ids.get(token, self.token_ids[self.UNK]))

        return utterance_ids

    def dialogs_to_ids(self, data):
        data_ids = []

        for dialog in data:
            dialog_ids = []

            for utterance in dialog:
                dialog_ids.append(self.utterance_to_ids(utterance))
            data_ids.append(dialog_ids)

        return data_ids

    def ids_to_tokens(self, ids):
        padding_id = self.token_ids[self.PAD]
        return [self.vocabulary[id] for id in ids if id != padding_id]

    def token_to_id(self, token):
        return self.token_ids[token]

    def get_train_dataset(self, context_size=2, min_reply_length=None, max_reply_length=None):
        return self.get_dataset(self.train_dialogs, context_size, min_reply_length, max_reply_length)

    def get_validation_dataset(self, context_size=2, min_reply_length=None, max_reply_length=None):
        return self.get_dataset(self.validation_dialogs, context_size, min_reply_length, max_reply_length)

    def get_test_dataset(self, context_size=2, min_reply_length=None, max_reply_length=None):
        return self.get_dataset(self.test_dialogs, context_size, min_reply_length, max_reply_length)

    def get_dataset(self, dialogs, context_size, min_reply_length, max_reply_length):
        dialogs_ids = self.dialogs_to_ids(dialogs)
        return DPDataset(self, dialogs_ids, context_size, min_reply_length, max_reply_length)

    def get_collator(self, reply_length = None):
        return DPCollator(self.token_ids[self.PAD], reply_length=reply_length)