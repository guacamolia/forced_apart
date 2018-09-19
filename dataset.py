import numpy as np
import torch
from torch.utils.data import Dataset

from utils import encode_sentence, restore_weights, cuda, match_embeddings


class MimicDataset(Dataset):
    def __init__(self, df, voc, autoencoder, max_len=600):
        super(MimicDataset, self).__init__()

        self.data = df['SENTENCES']
        self.labels = df['HEART']

        self.max_len = max_len

        self.w2idx = voc.w2idx
        self.idx2w = voc.idx2w
        self.autoencoder = autoencoder
        self.autoencoder_dim = autoencoder.hidden_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ehr = self.data.iloc[idx]
        sentences = []
        for sentence in ehr:
            ids = encode_sentence(sentence, self.w2idx, 20)
            sentences.append(ids)

        sentences = self.autoencoder.encoder(torch.Tensor(sentences).long().cuda()).cpu()

        if sentences.shape[0] >= self.max_len:
            sentences = sentences[:self.max_len]
        else:
            to_append = np.ones((self.max_len - sentences.shape[0], self.autoencoder_dim))
            to_append *= self.w2idx["<pad>"]
            sentences = torch.cat((sentences, torch.FloatTensor(to_append)), dim=0)

        icd = int(self.labels.iloc[idx])

        return sentences, icd
