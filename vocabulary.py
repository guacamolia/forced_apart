import itertools
import warnings
import pickle

from preprocess import read_pickled_notes


class Vocabulary:
    def __init__(self, text):
        if isinstance(text, str):
            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer('\w+')
            self.tokens = set(tokenizer.tokenize(text.lower()))
            warnings.warn("No words are filtered. Basic RegexpTokenizer is used for tokenization. Punctuation is ignored")
        elif isinstance(text, list):
            self.tokens = set(itertools.chain.from_iterable(text))
        elif isinstance(text, set):
            self.tokens = text
        else:
            raise TypeError('Invalid type of input data. Acceptable types are string, list and set.')

        self.w2idx = {word: idx+1 for idx, word in enumerate(self.tokens)}
        self.idx2w = {idx: word for word, idx in self.w2idx.items()}
        self.w2idx['<pad>'] = 0
        self.idx2w[0] = '<pad>'
        assert len(self.w2idx) == len(self.idx2w)

    def add_token(self, token):
        if token in self.w2idx.keys():
            print('Token "{}" is already present in the vocabulary'.format(token))
            return self
        cur_len = self.get_length()
        self.w2idx[token] = cur_len
        self.idx2w[cur_len] = token

        assert len(self.w2idx) == len(self.idx2w)
        return self

    def get_length(self):
        return len(self.w2idx)

    def save(self, loc):
        with open(loc, 'wb') as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    target_loc = 'data/dataframe.pkl'
    patients = read_pickled_notes(target_loc)
    voc = Vocabulary(patients['TOKENS'].tolist())
    voc.add_token('<end>')
    with open('data/voc.pkl', 'wb') as f:
        pickle.dump(voc.w2idx, f)
    print('Done')

