from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer
from torch.utils.data import Dataset


class IMDB(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.s = sentences
        self.l = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.s[idx]
        label = self.l[idx]
        if type(self.tokenizer) is BertTokenizer:
            token_result = self.tokenizer(text, return_tensors="pt")
        else:
            token_result = self.tokenizer(text)
        return token_result, label

