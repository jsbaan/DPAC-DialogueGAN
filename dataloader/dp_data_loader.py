from torch.utils.data.dataloader import DataLoader
from .dp_corpus import DPCorpus

class DPDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64):
        if dataset == None:
            corpus = DPCorpus(vocabulary_limit=5000)
            dataset = corpus.get_train_dataset(2, 5, 20)

        collator = dataset.corpus.get_collator(reply_length=20)

        super().__init__(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, drop_last=True)