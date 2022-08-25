import functools
import operator

import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

from preprocessing import *

device = torch.device('cuda')
BIN_NUM = 100
BATCH_SIZE = 150


class Review:
    def __init__(self, sent, sent_tok, label) -> None:
        self.sent = sent
        self.tok = sent_tok
        self.label = label


def download(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(texts, model, tokenizer, language) -> list:
    formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
    original_texts = [formatter_fn(txt) for txt in texts]
    tokens = tokenizer(original_texts, return_tensors='pt', padding='max_length', truncation=True).to(device)
    translated = model.generate(**tokens)
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts


def back_translate(texts, language_src, language_dst):
    translated = translate(texts, en2other_model, en2other_tokenizer, language_dst)
    back_translated = translate(translated, other2en_model, other2en_tokenizer, language_src)
    return back_translated


def binning(trains: Review):
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
    # en -> others
    en2other_tokenizer, en2other_model = download('Helsinki-NLP/opus-mt-en-ROMANCE')
    # other -> en
    other2en_tokenizer, other2en_model = download('Helsinki-NLP/opus-mt-ROMANCE-en')
    en2other_model.to(device)
    en2other_model.eval()
    other2en_model.to(device)
    other2en_model.eval()
    trains = []
    for s, l in zip(data['train_sentences'], data['train_labels']):
        tok = en2other_tokenizer(s)['input_ids']
        trains.append(Review(s, tok, l))
    bins = binning(trains)
    results = {'cleaned': [], 'label': []}
    for b in tqdm(bins):
        # Latin, French, Spanish, Italian
        for des in ['la', 'fr', 'es', 'it']:
            toks = [review.sent for review in b]
            labels = [review.label for review in b]
            with torch.no_grad():
                texts = back_translate(toks, 'en', des)
            results['cleaned'].append(texts)
            results['label'].append(labels)
    results['cleaned'] = functools.reduce(operator.iconcat, results['cleaned'], [])
    results['label'] = functools.reduce(operator.iconcat, results['label'], [])
    df = pd.DataFrame.from_dict(results)
    df.to_csv('./back_trans.csv', index=False)
