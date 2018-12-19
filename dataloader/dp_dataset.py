from torch.utils.data.dataset import Dataset
from torch import LongTensor

class DPDataset(Dataset):
    def __init__(self, corpus, dialogs, context_size=2, min_reply_length=None, max_reply_length=None):
        self.corpus = corpus

        self.contexts = []
        self.replies = []
        for dialog in dialogs:
            max_start_i = len(dialog) - context_size
            for start_i in range(max_start_i):
                reply = dialog[start_i + context_size]
                context = []
                for i in range(start_i, start_i+context_size):
                    context.extend(dialog[i])

                if (min_reply_length is None or len(reply) >= min_reply_length) and \
                        (max_reply_length is None or len(reply) <= max_reply_length):
                    self.contexts.append(context)
                    self.replies.append(reply)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, item):
        context = self.contexts[item]
        replies = self.replies[item]

        return (LongTensor(context), LongTensor(replies))