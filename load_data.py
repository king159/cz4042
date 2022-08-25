from collections import Counter, OrderedDict
from itertools import product
from typing import Dict, List, Any

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab
import pandas as pd

from preprocessing import split_dataset

VOCAB_SIZE = 6000


class IMDB(Dataset):
    def __init__(self, data, tokenizer=None):
        self.s = data['sent']
        self.l = data['label']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        text = self.s[idx]
        label = self.l[idx]
        if self.tokenizer:
            text = self.tokenizer(text)
        return text, label


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(torch.tensor(_text))
    label_list = torch.tensor(label_list).float()
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    return text_list, label_list


def create_dataloader(mode: str, batch_size: int, csv_file: str) -> Dict[str, DataLoader]:
    data_dict = split_dataset(pd.read_csv("./data/%s" % csv_file), mode='full')
    loaders = None
    if mode != 'bert':
        tokenizer = get_tokenizer("basic_english")
        counter = Counter()
        for i in (data_dict['train']['sent']):
            counter.update(tokenizer(i))
        ordered_dict = OrderedDict(counter.most_common(VOCAB_SIZE - 1))
        vocabulary = vocab(ordered_dict)
        # set <unk> as the last token
        vocabulary.set_default_index(VOCAB_SIZE - 1)
        text_pipeline = lambda x: vocabulary(tokenizer(x))
        loaders = {x: DataLoader(IMDB(data_dict[x], text_pipeline), batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_batch)
                   for x in ['train', 'val', 'test']}
    elif mode == 'bert':
        # we delay the tokenization as bert has it own tokenizer
        loaders = {x: DataLoader(IMDB(data_dict[x]), batch_size=batch_size, shuffle=True)
                   for x in ['train', 'val', 'test']}
    return loaders


def create_optimizer(name: str, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)


def load_hyperparameters(name: str) -> List[Dict[str, Any]]:
    with open('hyper_para.yaml', 'r') as f:
        hyper_para = yaml.safe_load(f)[name]
    result = []
    for i in list(product(*hyper_para.values())):
        result.append({j: i for j, i in zip(hyper_para.keys(), i)})
    return result


if __name__ == '__main__':
    print(load_hyperparameters('cnn'))
