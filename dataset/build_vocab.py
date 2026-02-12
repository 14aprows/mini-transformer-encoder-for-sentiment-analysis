from torchtext.datasets import AG_NEWS
from collections import Counter
import torch
import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text.split()

def build_vocal(max_vocab=20000):
    counter = Counter()

    train_iter = AG_NEWS(split="train")

    for label, text in train_iter:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {"<unk>": 1, "<pad>": 0}

    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    return vocab