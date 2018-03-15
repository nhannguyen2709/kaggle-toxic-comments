import numpy as np
import pandas as pd
import os

from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import roc_auc_score


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
        tokenizer.fit_on_texts(train_texts + test_texts)
        
        train_sequences = tokenizer.texts_to_sequences(train_texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)

        x_train = pad_sequences(train_sequences, maxlen=maxlen)
        x_test = pad_sequences(test_sequences, maxlen=maxlen)
        
        word_index = tokenizer.word_index
        print('Found %s unique tokens in corpus.' % len(word_index))
        return x_train, x_test, word_index


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
