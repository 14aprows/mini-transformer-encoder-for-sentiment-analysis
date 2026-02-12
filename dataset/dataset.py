import torch 
from torch.utils.data import Dataset
from dataset import load_dataset
from .build_vocab import tokenize

class AGNewsDataset(Dataset):
    def __init__(self, vocab, split="train", max_len=64, limit=10000):
        dataset = load_dataset("ag_news", split=split)

        self.samples = []
        self.vocab = vocab
        self.max_len = max_len

        for i, item in enumerate(dataset):
            if i >= limit:
                break

            tokens = tokenize(item["text"])
            ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
            ids = ids[:max_len]

            pad_len = max_len - len(ids)
            ids += [vocab["<pad>"]] * pad_len

            self.samples.append((torch.tensor(ids), item["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]