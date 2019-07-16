import random
import tensorflow as tf
import numpy as np
import pickle
import os

class Vocab:

    def __init__(self, w2idx, idx2w, vocab_size, maxlen, org_vocab_size):

        self.w2idx = w2idx
        self.idx2w = idx2w
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.org_vocab_size = org_vocab_size

class Dataset:

    def __init__(self, path1, path2, batch_size, vocab1_max=10**6, vocab2_max=10**6, length_limit=10**3, sample=None, vocab=None, reverse_source=False, easy_subset = None, shuffle=True, mono=None):

        X, Y = self.load(path1), self.load(path2)

        X, Y = self.remove_too_long(X, Y, length_limit)

        if reverse_source:
            X = [list(reversed(x)) for x in X]

        assert not (sample and easy_subset)
        if sample is not None:
            X, Y = self.sample(X, Y, sample)
        elif easy_subset is not None:
            X, Y = self.easy_subset(X, Y, easy_subset)

        if vocab is None:
            X, first = self.to_ints(X, vocab1_max)
            Y, second = self.to_ints(Y, vocab2_max)
        else:
            if isinstance(vocab, str):
                vocab = self.load_vocab(vocab)

            X, first = self.from_vocab(X, vocab[0])
            Y, second = self.from_vocab(Y, vocab[1])
        
        X_lengths = list(map(len, X))
        Y_lengths = list(map(len, Y))

        ids = list(range(len(X)))

        X = self.pad(X, first.maxlen)
        Y = self.pad(Y, second.maxlen)

        tf_dataset, num_batches = self.get_dataset(X, X_lengths, Y, Y_lengths, ids, batch_size, shuffle)

        self.X, self.Y, self.X_lengths, self.Y_lengths, self.ids = X, Y, X_lengths, Y_lengths, ids
        self.first, self.second = first, second

        self.tf_dataset, self.batch_size, self.num_batches = tf_dataset, batch_size, num_batches
        self.reverse_source = reverse_source

        if mono is not None:
            offset1, bind_hard = mono
            offset2 = offset1 if not bind_hard else offset1 + len(X)
        else:
            offset1, offset2 = 0, 0
        self.offset1, self.offset2 = offset1, offset2


    def load(self, path):

        X = open(path, "r", encoding = "utf-8").readlines()

        X = [x.strip().split(" ") for x in X]

        return X

    def remove_too_long(self, X, Y, length_limit):
        inds = [i for i in range(len(X)) if len(X[i]) <= length_limit and len(Y[i]) <= length_limit]

        return [X[i] for i in inds], [Y[i] for i in inds]

    def sample(self, X, Y, sample):

        N = len(X)

        if 0 <= sample <= 1:
            # Interpret as ratio
            assert type(sample) == float
            k = round(N * sample)
        else:
            # Interpret as count
            assert type(sample) == int
            k = sample

        indices = random.sample(list(range(N)), k)

        return [X[i] for i in indices], [Y[i] for i in indices]


    def easy_subset(self, D1, D2, n):
        print("USING EASY SUBSET")
        c = {}
        for d in D2:
            for w in d:
                c[w] = c.get(w, 0) + 1

        inds = np.array(sorted(range(len(D1)), key = lambda i : min(c[w] for w in D2[i]), reverse=True)[:n])
        return np.take(D1, inds), np.take(D2, inds)

    def to_ints(self, X, max_vocab):

        counts = {}

        for x in X:
            for w in x:
                counts[w] = counts.get(w, 0) + 1

        w2idx = {x[0]:4+i for i, x in enumerate(sorted(counts.items(), reverse=True, key = lambda p : p[1])[:max_vocab])}

        w2idx["<PAD>"] = 0
        w2idx["<START>"] = 1
        w2idx["<END>"] = 2
        w2idx["<UNK>"] = 3

        idx2w = {v:k for k,v in w2idx.items()}

        X = [[w2idx["<START>"]] + [w2idx.get(w, w2idx["<UNK>"]) for w in x] + [w2idx["<END>"]] for x in X]

        maxlen = max(map(len, X))

        return X, Vocab(w2idx, idx2w, len(w2idx), maxlen, len(counts))

    def from_vocab(self, X, vocab):

        X = [[vocab.w2idx["<START>"]] + [vocab.w2idx.get(w, vocab.w2idx["<UNK>"]) for w in x] + [vocab.w2idx["<END>"]] for x in X]

        return X, vocab


    def pad(self, X, maxlen):

        return tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen, padding="post")

    def get_dataset(self, X, X_lengths, Y, Y_lengths, ids, batch_size, shuffle):

        dataset = tf.data.Dataset.from_tensor_slices((X, X_lengths, Y, Y_lengths, ids))
        if shuffle:
            dataset = dataset.shuffle(len(X))

        dataset = dataset.batch(batch_size, drop_remainder = True)

        num_batches = len(X) // batch_size

        return dataset, num_batches

    def save_vocab(self, vocab_path):

        if os.path.isfile(vocab_path):
            print("Can't save vocab. File already exists")
            exit(1)

        with open(vocab_path, "wb") as f:
            pickle.dump((self.first, self.second), f)

    def load_vocab(self, vocab_path):

        if not os.path.isfile(vocab_path):
            print("Can't load vocab. File doesn't exist.")
            exit(1)

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        return vocab


    def __iter__(self):
        return iter(self.tf_dataset)


    def summarize(self):
        print("Source language:")
        print("  Num sentences: {} ({} batches)".format(len(self.X), self.num_batches))
        print("  Num words:", sum(self.X_lengths))
        unks = sum(1 for x in self.X for w in x if w == self.first.w2idx["<UNK>"])
        print("  Num UNKS: {} ({:.1f} per sentence)".format(unks, unks/len(self.X)))
        print("  Vocab size: {} (original {})".format(self.first.vocab_size, self.first.org_vocab_size))
        print("  Longest:", self.first.maxlen)
        print("  Reversed:", self.reverse_source)

        print("\nTarget language:")
        print("  Num sentences: {} ({} batches)".format(len(self.Y), self.num_batches))
        print("  Num words:", sum(self.Y_lengths))
        unks = sum(1 for x in self.Y for w in x if w == self.second.w2idx["<UNK>"])
        print("  Num UNKS: {} ({:.1f} per sentence)".format(unks, unks/len(self.X)))
        print("  Vocab size: {} (original {})".format(self.second.vocab_size, self.second.org_vocab_size))
        print("  Longest:", self.second.maxlen)
