import torch 
from torch.utils.data import Dataset
from torchtext.datasets import AG_NEWS
from .build_vocab import tokenize

class AGNewsDataset(Dataset):
    def __init__(self, vocab, max_len=64, split="train", limit=10000):
        self.samples = []
        self.vocab = vocab
        self.max_len = max_len

        data_iter = AG_NEWS(split=split)

        for i, (label, text) in enumerate(data_iter):
            if i >= limit:
                break

            tokens = tokenize(text)
            ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
            ids = ids[:max_len]

            pad_len = max_len - len(ids)
            ids = ids + [vocab["<pad>"]] * pad_len

            self.samples.append((torch.tensor(ids), label-1))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]