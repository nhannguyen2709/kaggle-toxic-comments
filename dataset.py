import numpy as np
import pandas as pd
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class ToxicCommentsDataset:
    """Load and process the dataset for scikit-learn and keras models."""
    def __init__(self, data_dir, train_csv_file, test_csv_file):
        self.data_dir = data_dir
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
    
    def get_texts_and_train_labels(self):
        train = pd.read_csv(os.path.join(self.data_dir, self.train_csv_file))
        test = pd.read_csv(os.path.join(self.data_dir, self.test_csv_file))
        train_comment_text = train["comment_text"].str.lower()
        test_comment_text = test["comment_text"].str.lower()
        train_texts = list(train_comment_text.values)
        test_texts = list(test_comment_text.values)
        y_train = np.array(train.loc[:, train.columns[2:]])
        return train_texts, y_train, test_texts
    
    def tokenize_by_keras(self, max_words, maxlen):
        train_texts, _, test_texts = self.get_texts_and_train_labels()   
        tokenizer = Tokenizer(num_words=max_words, lower=True)
        tokenizer.fit_on_texts(train_texts)
        train_sequences = tokenizer.texts_to_sequences(train_texts)
        x_train = pad_sequences(train_sequences, maxlen=maxlen)
        train_word_index = tokenizer.word_index
        print('Found %s unique tokens in train corpus.' % len(train_word_index))
        tokenizer.fit_on_texts(test_texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)
        x_test = pad_sequences(test_sequences, maxlen=maxlen)
        test_word_index = tokenizer.word_index
        print('Found %s unique tokens in test corpus.' % len(test_word_index))
        return x_train, x_test
    
    































