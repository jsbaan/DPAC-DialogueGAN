from torch.utils.data.dataset import Dataset

class DPDataset(Dataset):
    def __init__(self, corpus, dialogs, context_size):
        self.corpus = corpus

        self.contexts = []
        self.replies = []
        for dialog in dialogs:
            max_start_i = len(dialog) - context_size - 1
            for start_i in range(max_start_i):
                context = []
                for i in range(start_i, start_i+context_size):
                    context.extend(dialog[i])

                self.contexts.append(context)
                self.replies.append(dialog[start_i + context_size + 1])

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, item):
        context = self.contexts[item]
        replies = self.replies[item]

        return (context, replies)