import nltk

class DailyDialogParser:
    def __init__(self, path, sos, eos, eou):
        self.path = path
        self.sos = sos
        self.eos = eos
        self.eou = eou

    def get_dialogs(self):
        train_dialogs = self.process_file(self.path + 'train.txt')
        validation_dialogs = self.process_file(self.path + 'validation.txt')
        test_dialogs = self.process_file(self.path + 'test.txt')
        return train_dialogs, validation_dialogs, test_dialogs

    def process_file(self, path):
        with open(path, 'r') as f:
            data = f.readlines()

        print("Parsing", path)
        return [self.process_raw_dialog(line) for line in data]

    def process_raw_dialog(self, raw_dialog):
        raw_utterances = raw_dialog.split('__eou__')
        return [self.process_raw_utterance(raw_utterance) for raw_utterance in raw_utterances if not raw_utterance.isspace()]

    def process_raw_utterance(self, raw_utterance):
        raw_sentences = nltk.sent_tokenize(raw_utterance)

        utterence = []
        for raw_sentence in raw_sentences:
            utterence.extend(self.process_raw_sentence(raw_sentence))

        return utterence + [self.eou]

    def process_raw_sentence(self, raw_sentence):
        raw_sentence = raw_sentence.lower()
        raw_sentence = raw_sentence.split()
        return [self.sos] + raw_sentence + [self.eos]