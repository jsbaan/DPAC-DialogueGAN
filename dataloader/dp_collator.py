class DPCollator:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        contexts, replies = zip(*batch)

        padded_contexts = self.pad(contexts)
        padded_replies = self.pad(replies)

        return padded_contexts, padded_replies

    def pad(self, data):
        max_length = max([len(row) for row in data])

        padded_data = []
        for row in data:
            padding = [self.pad_token] * (max_length-len(row))
            padded_data.append(row + padding)

        return padded_data