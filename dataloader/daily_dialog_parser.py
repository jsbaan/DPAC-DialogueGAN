import nltk

class DailyDialogParser:
    SOS = '<s>' # Start of sentence token
    EOS = '</s>' # End of sentence token
    EOU = '</u>' # End of utterance token

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

        return utterence + [self.EOU]

    def process_raw_sentence(self, raw_sentence):
        raw_sentence = raw_sentence.lower()
        raw_sentence = raw_sentence.split()
        return [self.SOS] + raw_sentence + [self.EOS]