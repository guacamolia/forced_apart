import logging

import numpy as np
import torch

from nltk.tokenize import RegexpTokenizer
from torch.autograd import Variable
from tqdm import tqdm


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking)
    lengths = masks.sum(dim=dim)
    return lengths


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def pad_sequence(sequence, token, max_len):
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    elif len(sequence) < max_len:
        sequence = sequence + [token]*(max_len - len(sequence))
    return sequence


def save_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), filename)
    logging.info('Model saved: {os.path.basename(filename)}')


def restore_weights(model, filename):

    if not isinstance(filename, str):
        filename = str(filename)
    map_location = None
    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage
    state_dict = torch.load(filename, map_location=map_location)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(state_dict)
    logging.info('Model restored: {os.path.basename(filename)}')
    return


def encode_sentence(sentence, w2idx, max_len):
    enc_sentence = seq2idx(sentence, w2idx)
    enc_sentence = enc_sentence + [w2idx['<end>']]
    enc_sentence = pad_sequence(enc_sentence, w2idx['<pad>'], max_len)
    enc_sentence = np.array(enc_sentence)
    return enc_sentence


def seq2idx(words, voc, tokenizer=RegexpTokenizer(r'\w+')):
    if isinstance(words, str):
        words = tokenizer.tokenize(words)

    idx = []
    for word in words:
        if word in voc.keys():
            idx.append(voc[word])
        else:
            idx.append(voc['<unk>'])
    return idx


def match_embeddings(idx2w, w2vec, dim):
    embeddings = []
    voc_size = len(idx2w)
    for idx in tqdm(range(voc_size)):
        word = idx2w[idx]
        if word not in w2vec:
            embeddings.append(np.random.uniform(low=-1.2, high=1.2, size=(dim, )))
        else:
            embeddings.append(w2vec[word])

    embeddings = np.stack(embeddings)
    return embeddings


def read_embeddings(datafile):
    w2vec = {}
    with open(datafile, 'r') as f:
        f.readline()
        for line in f.readlines():
            word = line.split()[0]
            vec = np.array([float(num) for num in line.split()[1:]])
            w2vec[word] = vec
    return w2vec