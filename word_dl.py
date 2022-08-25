from transformers import RobertaTokenizer, RobertaForMaskedLM
import functools
import operator
from typing import List
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from preprocessing import split_dataset

device = torch.device('cuda')
BIN_NUM = 100
BATCH_SIZE = 200
MODEL_LENGTH = 512


def chunk_max(tok, max_length=MODEL_LENGTH):
    if tok.size(1) > max_length:
        tok = torch.cat([tok[:, :(max_length - 1)], tok[:, -1:]], dim=1)
        return tok
    else:
        return tok


class Review:
    def __init__(self, sent, sent_tok, label) -> None:
        self.sent = sent
        self.tok = sent_tok
        self.label = label


def get_text(texts, model, tokenizer) -> List[str]:
    input_ids = torch.tensor(tokenizer.batch_encode_plus(texts, padding=True)['input_ids'])
    input_ids = chunk_max(input_ids)
    mask_idx = np.random.choice(range(1, input_ids.size(1)), int(input_ids.size(1)/5), replace=False)
    input_ids[:, mask_idx] = tokenizer.mask_token_id
    input_ids = input_ids.to(device)
    predict = model(input_ids)['logits']
    replace = predict[:, mask_idx, :].topk(top_k).indices
    results = []
    for i in range(top_k):
        temp = input_ids.clone()
        temp[:, mask_idx] = replace[:, :, i]
        results.append(tokenizer.batch_decode(temp, skip_special_tokens=True))
    results = functools.reduce(operator.iconcat, results, [])
    return results


def binning(trains: List[Review]) -> List[List[Review]]:
    trains_sorted = sorted(trains, key=lambda x: len(x.tok))
    lengths = [len(x.tok) for x in trains_sorted]
    histogram = np.histogram(lengths, bins=BIN_NUM)
    groups = np.digitize(lengths, histogram[1], right=True)
    bins = []
    for i in range(BIN_NUM + 1):
        bins.append([])
    for num, j in enumerate(trains_sorted):
        bins[groups[num]].append(j)
    bins = sorted(list(filter(None, bins)), key=lambda x: len(x), reverse=True)
    result = []
    for b in bins:
        if len(b) > BATCH_SIZE:
            for x in range(0, len(b), BATCH_SIZE):
                result.append(b[x:x + BATCH_SIZE])
        else:
            result.append(b)
    return result


if __name__ == '__main__':
    top_k = 5
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
    model.eval()
    df = pd.read_csv('./data/IMDB_cleaned.csv')
    data = split_dataset(df, 'small')
    trains = []
    for s, l in zip(data['train_sentences'], data['train_labels']):
        tok = tokenizer.encode(s)
        trains.append(Review(s, tok, l))
    bins = binning(trains)
    results = {'cleaned': [], 'label': []}
    for b in tqdm(bins):
        sents = [review.sent for review in b]
        labels = [review.label for review in b]
        with torch.no_grad():
            texts = get_text(sents, model, tokenizer)
        labels = np.repeat(labels, 5).tolist()
        results['cleaned'].append(texts)
        results['label'].append(labels)
    results['cleaned'] = functools.reduce(operator.iconcat, results['cleaned'], [])
    results['label'] = functools.reduce(operator.iconcat, results['label'], [])
    df = pd.DataFrame.from_dict(results)
    df.to_csv('./roberta_word.csv', index=False)
