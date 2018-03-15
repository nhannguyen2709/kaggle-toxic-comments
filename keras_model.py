import h5py
import numpy as np
import pandas as pd
import os

from utils import ToxicCommentsDataset, RocAucEvaluation

# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

# laptop_data_dir = '/mnt/d/kaggle-toxic-comments/data'
desktop_data_dir = '/home/nhan/Downloads/toxic_comments'
embedding_dir = '/home/nhan/Downloads/word_embeddings/'

toxic_comments_dataset = ToxicCommentsDataset(desktop_data_dir,
                                              'train.csv',
                                              'test.csv')
max_words = 100000
maxlen = 150
_, y_train, _ = toxic_comments_dataset.get_texts_and_train_labels()
x_train, x_test, word_index = toxic_comments_dataset.tokenize_by_keras(max_words=max_words, maxlen=maxlen)

embedding_files = sorted(os.listdir(embedding_dir))[:2]
list_embeddings_index = []
for file in embedding_files:
    embeddings_index = {}
    with open(os.path.join(embedding_dir, file),encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    list_embeddings_index.append(embeddings_index)

num_words = min(max_words, len(word_index) + 1)
embedding_matrix0 = np.zeros((num_words, 300))
for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = list_embeddings_index[0].get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix0[i] = embedding_vector

num_words = min(max_words, len(word_index) + 1)
embedding_matrix1 = np.zeros((num_words, 200))
for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = list_embeddings_index[1].get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix1[i] = embedding_vector

del list_embeddings_index

sequence_input = Input(shape=(maxlen, ))
x_glove300 = Embedding(max_words, 300, weights=[embedding_matrix0], trainable = False)(sequence_input)
x_glove200 = Embedding(max_words, 200, weights=[embedding_matrix1], trainable = False)(sequence_input)
x = concatenate([x_glove300, x_glove200])
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
x = Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
outputs = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, outputs)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])
model.summary()

batch_size = 256
epochs = 15
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.9, random_state=233)

filepath = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
roc_auc = RocAucEvaluation(validation_data=(x_validation, y_validation), interval = 1)
callbacks_list = [roc_auc,checkpoint, early]
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs,
          validation_data=(x_validation, y_validation),
          callbacks = callbacks_list,verbose=1)
