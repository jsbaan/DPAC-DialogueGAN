from torch import LongTensor

class DPCollator:
    def __init__(self, pad_token, reply_length=None):
        self.pad_token = pad_token
        self.reply_length = reply_length

    def __call__(self, batch):
        contexts, replies = zip(*batch)

        padded_contexts = self.pad(contexts)
        padded_replies = self.pad(replies, self.reply_length)

        return padded_contexts, padded_replies

    def pad(self, data, length = None):
        max_length = length
        if max_length is None:
            max_length = max([len(row) for row in data])

        padded_data = []
        for row in data:
            padding = [self.pad_token] * (max_length-len(row))
            padded_data.append(list(row) + padding)

        return LongTensor(padded_data)