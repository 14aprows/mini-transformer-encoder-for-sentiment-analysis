import re 
from collections import Counter

def tokenize(text): 
    text = text.lower() 
    text = re.sub(r"[^a-z0-9\s]", "", text) 
    return text.split()

def build_vocab(texts, max_vocab=20000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    vocab = {"<pad>": 0, "<unk>": 1}

    for word, _ in counter.most_common(max_vocab-2):
        vocab[word] = len(vocab)
        
    return vocab