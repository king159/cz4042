import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer

# nltk.download('stopwords')

SEED = 1946614
random.seed(SEED)


def clean_dataset(old_df: pd.DataFrame, save=False):
    new_df = old_df.copy()
    lowercase_review = [i.lower() for i in new_df['review'].tolist()]
    new_df['cleaned'] = lowercase_review
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')

    def _remove_stopwords(text):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        final_tokens = [token for token in tokens if token not in stopword_list]
        cleaned_text = ' '.join(final_tokens)
        return cleaned_text

    new_df['cleaned'] = new_df['cleaned'].apply(_remove_stopwords)

    def _remove_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    new_df['cleaned'] = new_df['cleaned'].apply(_remove_html)

    def _remove_brackets(text):
        return re.sub(r'\[[^]]*', '', text)

    new_df['cleaned'] = new_df['cleaned'].apply(_remove_brackets)
    new_df['label'] = new_df['sentiment'].astype('category').cat.codes
    if save:
        new_df.to_csv('data/cleaned_data.csv', index=False)
    else:
        return new_df


def split_dataset(df: pd.DataFrame, mode='full') -> Dict[str, Dict[str, List[Any]]]:
    data = [(r, label) for r, label in zip(df['cleaned'], df['label'])]
    p_data = [d for d in data if d[1] == 1]
    n_data = [d for d in data if d[1] == 0]
    # ensure that the positive and negative data are exactly balanced after splitting
    for d in ['p_data', 'n_data']:
        data = locals()[d]
        random.shuffle(data)
        datasize = len(data)
        locals()[d + '_dict'] = defaultdict()
        if mode == 'full':
            # 35000 train, 7500 dev, 7500 test
            locals()[d + '_dict']['train'], locals()[d + '_dict']['val'], locals()[d + '_dict']['test'] = data[0:int(
                0.7 * datasize)], data[int(0.7 * datasize):int(
                0.85 * datasize)], data[int(0.85 * datasize):]
        elif mode == 'small':
            # 7000 train, 7500 dev, 7500 test
            locals()[d + '_dict']['train'], locals()[d + '_dict']['val'], locals()[d + '_dict'][
                'test'] = random.choices(data[0:int(0.7 * datasize)],
                                         k=int(0.7 * datasize / 5)), \
                          data[int(0.7 * datasize):int(0.85 * datasize)], \
                          data[int(0.85 * datasize):]
    data2 = defaultdict()
    for x in ['train', 'val', 'test']:
        data2[x] = []
    for d in ['p_data_dict', 'n_data_dict']:
        for x in ['train', 'val', 'test']:
            data2[x] += locals()[d][x]
    outputs = {}
    for x in ['train', 'val', 'test']:
        outputs[x] = {"sent": [d[0] for d in data2[x]], "label": [d[1] for d in data2[x]]}
    return outputs
