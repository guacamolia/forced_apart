import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from autoencoder.autoencoder import Autoencoder
from dataset import MimicDataset
from model import AttentionModel
from preprocess import read_pickled_notes
from utils import cuda, variable, match_embeddings, restore_weights


def train():
    # PARAMETERS #
    batch_size = 5
    nb_epochs = 100
    hidden_size = 128

    # DATA FILES #
    patients_loc = 'data/dataframe.pkl'
    patients = read_pickled_notes(patients_loc)
    w2vec_file = 'autoencoder/data/w2vec.pkl'
    voc_file = 'autoencoder/data/vocabulary.pkl'
    model_file = 'autoencoder/saved_models/autoencoder.pt'

    # VOCABULARY #
    with open(voc_file, 'rb') as f:
        voc = pickle.load(f)
    w2idx = voc.w2idx
    idx2w = voc.idx2w
    init_idx = w2idx['<start>']
    padding_idx = w2idx['<pad>']
    voc_size = voc.get_length()

    # PRETRAINED AUTOENCODER LOAD #
    with open(w2vec_file, 'rb') as f:
        w2vec = pickle.load(f)
    embedding_dim = len(random.choice(list(w2vec.values())))
    embeddings = torch.Tensor(match_embeddings(idx2w, w2vec, dim=embedding_dim))
    autoencoder = cuda(Autoencoder(128, voc_size, padding_idx, init_idx, 20, embeddings))
    restore_weights(autoencoder, model_file)

    # DATASET
    dataset = MimicDataset(patients, voc, autoencoder)

    # Creating data indices for training and validation splits:
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, dev_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=dev_sampler)

    datasets = {'train': train_loader, 'dev': dev_loader}

    # MODEL
    model = cuda(nn.DataParallel(AttentionModel(n_classes=3, input_size=autoencoder.hidden_size, hidden_size=hidden_size)))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, nb_epochs + 1):
        total = 0
        correct = 0
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            for item in tqdm(datasets[phase]):
                optimizer.zero_grad()

                records, icds = item
                records, icds = variable(records), variable(icds)

                outs = model(records)
                loss = criterion(outs, icds)

                if phase == 'dev':
                    _, predictions = torch.max(outs, dim=1)
                    correct += torch.sum(torch.eq(predictions, icds), dim=0)
                    total += records.shape[0]

                elif phase == 'train':
                    loss.backward()
                    optimizer.step()

            if epoch % 1 == 0:
                print('_________________________________________')
                print('Epoch #{}'.format(epoch))
                if phase == 'dev':
                    accuracy = correct.item() * 1.0 / total
                    print('DEV  \t loss: {:.4f}'.format(loss.item()))
                    print('Accuracy: {:.4f}'.format(accuracy))


if __name__ == '__main__':
    train()
    print()
