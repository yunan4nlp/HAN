from collections import Counter
import numpy as np
from data.Dataloader import *

class Vocab(object):
    PAD, UNK = 0, 1
    def __init__(self, word_counter, l1_counter, l2_counter, role_counter, min_occur_count = 2):
        self._id2word = ['<pad>', '<unk>']
        self._wordid2freq = [10000, 10000]
        self._id2extword = ['<pad>', '<unk>']
        self._id2role = ['<pad>', '<unk>']

        self._id2l1 = []
        self._id2l2 = []

        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for l, count in l1_counter.most_common():
            self._id2l1.append(l)

        for l, count in l2_counter.most_common():
            self._id2l2.append(l)

        for role, count in role_counter.most_common():
            self._id2role.append(role)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._l12id = reverse(self._id2l1)
        if len(self._l12id) != len(self._id2l1):
            print("serious bug: l1 dumplicated, please check!")

        self._l22id = reverse(self._id2l2)
        if len(self._l22id) != len(self._id2l2):
            print("serious bug: l2 dumplicated, please check!")

        self._role2id = reverse(self._id2role)
        if len(self._role2id) != len(self._id2role):
            print("serious bug: role dumplicated, please check!")

        print("Vocab info: #words %d" % (self.vocab_size))
        print("l1 info: #l1 %d" % (self.l1_size))
        print("l2 info: #l2 %d" % (self.l2_size))
        print("role info: #l2 %d" % (self.role_size))


    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        return embeddings


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def l12id(self, xs):
        if isinstance(xs, list):
            return [self._l12id.get(x, self.UNK) for x in xs]
        return self._l12id.get(xs, self.UNK)

    def id2l1(self, xs):
        if isinstance(xs, list):
            return [self._id2l1[x] for x in xs]
        return self._id2l1[xs]

    def l22id(self, xs):
        if isinstance(xs, list):
            return [self._l22id.get(x, self.UNK) for x in xs]
        return self._l22id.get(xs, self.UNK)

    def id2l2(self, xs):
        if isinstance(xs, list):
            return [self._id2l2[x] for x in xs]
        return self._id2l2[xs]

    def role2id(self, xs):
        if isinstance(xs, list):
            return [self._role2id.get(x, self.UNK) for x in xs]
        return self._role2id.get(xs, self.UNK)

    def id2role(self, xs):
        if isinstance(xs, list):
            return [self._id2role[x] for x in xs]
        return self._id2role[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def l1_size(self):
        return len(self._id2l1)

    @property
    def l2_size(self):
        return len(self._id2l2)

    @property
    def role_size(self):
        return len(self._id2role)

def creatVocab(config):
    corpusFile = config.train_file
    min_occur_count = config.min_occur_count
    data = read_corpus(corpusFile, config.max_sent_length, config.max_turn_length)

    word_counter = Counter()

    role_counter = Counter()
    for inst in data:
        assert len(inst) == 4
        for role in inst[1]:
            role_counter[role] += 1
        for sents in inst[2]:
            for word in sents:
                word_counter[word] += 1


    l1, l2 = build_labels()

    l1_counter = Counter()
    l2_counter = Counter()

    for l in l1:
        l1_counter[l] += 1

    for l in l2:
        l2_counter[l] += 1

    return Vocab(word_counter, l1_counter, l2_counter, role_counter, min_occur_count)
