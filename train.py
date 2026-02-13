import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset.build_vocab import build_vocab
from dataset.dataset import AGNewsDataset
from models.transformer import TransformerClassifier
from trainer.trainer import Trainer


def main():

    raw_train = load_dataset("ag_news", split="train[:10000]")
    texts = [item["text"] for item in raw_train]

    vocab = build_vocab(texts)
    vocab_size = len(vocab)

    train_dataset = AGNewsDataset(vocab, split="train", limit=10000)
    val_dataset = AGNewsDataset(vocab, split="test", limit=2000)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=64
    )

    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(epochs=5)


if __name__ == "__main__":
    main()