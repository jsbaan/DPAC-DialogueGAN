import nltk

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'

class DailyDialogCorpus(object):
    def __init__(self, path='/daily_dialog/', max_utt_len=20, vocab_size=5000):
        self.tokenize = nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize

        self.train_corpus = self.process_file(path + 'daily_dialog_train.txt')
        self.validation_corpus = self.process_file(path + 'daily_dialog_validation.txt')
        self.test_corpus = self.process_file(path + 'daily_dialog_test.txt')

        self.build_vocab(vocab_size)
        print("Done loading corpus")

    def process_file(self, data)::
        with open(path, 'rb') as f:
            data = f.readlines()

        processed_data = []
        for raw_dialog in data:
            # What does this line do, convert unicode to ascii? TODO: check
            raw_dialog = raw_dialog.decode('ascii', 'ignore').encode()
            raw_dialog = raw_dialog.split('__eou__')

            dialog = []
            for sentence in raw_dialog:
                utterance = [BOS] + sentence.lower() + [EOS]

                all_lens.append(len(utterance))
                dialog.append(utterance)

            processed_data.append(dialog)

        return new_dialog

    def build_vocab(self, vocab_size):
        all_words = []
        for dialog in self.train_corpus:
            for sentence in dialog:
                all_words.extend(sentence)

        vocab_counter = Counter(all_words).most_common(vocab_size)

        self.vocab = [PAD, UNK] + list(vocab_counter.keys())
        self.reversed_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.reversed_vocab[UNK]

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