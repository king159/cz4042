# Text Sentiment Analysis: Exploring Data Augmentation

CZ4042 Neural Network & Deep Learning Group Project

## Abstraction and Conclusion

Deep learning-based studies of text sentiment analysis (TSA) usually desires a large 
labeled dataset in order to produce significant performance in a domain. However, such
demand in training data is often not satisfied given how expensive and tedious it is to
produce meaningful labeled text. To overcome the challenge of doing sentiment classification
on small dataset, data augmentation techniques are frequently used to obtain new
training texts through modifying the existing data. In this assignment, we aim to investigate
how data augmentation affects the training of deep learning models. With three
rule-based and three deep-learning-based augmentation methods applied on each of the
three deep language models chosen (i.e. CNN, BiLSTM, BERT), it is found that different
model responses differently on different augmentation methods.

## Project Architecture

### Data
The data is downloaded from
[IMDB movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/version/1), which is not included in the repository.

### Usage

#### Data preprocessing
In [preprocessing.py](preprocessing.py):
1. `clean_dataset` to create `IMDB_cleaned.csv`
2. `split_dataset` to split data into train/dev/test set

### Data augmentation (DL and rule-based)
Each will create one or serval csv as the augmented dataset
1. [word_dl.py](word_dl.py)
2. [sent_dl.py](sent_dl.py)
3. [back_trans.py](back_trans.py)
4. [rulebased_aug.ipynb](rulebased_aug.ipynb)