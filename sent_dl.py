import functools
import operator
from typing import List
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import pandas as pd
from preprocessing import split_dataset

device = torch.device('cuda')
BIN_NUM = 100
BATCH_SIZE = 30
MODEL_LENGTH = 1024


def chunk_max(tok, max_length=MODEL_LENGTH):
    if tok['input_ids'].size(1) > max_length:
        for i in tok.keys():
            tok[i] = torch.cat([tok[i][:, :(max_length - 1)], tok[i][:, -1:]], dim=1)
        return tok
    else:
        return tok


class Review:
    def __init__(self, sent, sent_tok, label) -> None:
        self.sent = sent
        self.tok = sent_tok
        self.label = label


def get_text(texts, model, tokenizer) -> list:
    tokens = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True).to(device)
    tok = chunk_max(tokens)
    tok = tok.to(device)
    generated = model.generate(**tok,
                               max_length=1000,
                               top_k=50,
                               do_sample=True,
                               no_repeat_ngram_size=2,
                               num_return_sequences=4)
    results = tokenizer.batch_decode(generated, skip_special_tokens=True)
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
    df = pd.read_csv('./data/IMDB_cleaned.csv')
    data = split_dataset(df, 'small')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    trains = []
    for s, l in zip(data['train_sentences'], data['train_labels']):
        tok = tokenizer.encode(s)
        trains.append(Review(s, tok, l))
    bins = binning(trains)
    results = {'cleaned': [], 'label': []}
    for b in tqdm(bins):
        sentences = [review.sent for review in b]
        labels = [review.label for review in b]
        results['cleaned'].append(sentences)
        results['label'].append(labels)
        with torch.no_grad():
            texts = get_text(sentences, model, tokenizer)
        labels = np.repeat(labels, 4).tolist()
        results['cleaned'].append(texts)
        results['label'].append(labels)
    results['cleaned'] = functools.reduce(operator.iconcat, results['cleaned'], [])
    results['label'] = functools.reduce(operator.iconcat, results['label'], [])
    df = pd.DataFrame.from_dict(results)
    df.to_csv('./gpt2_summary.csv', index=False)
