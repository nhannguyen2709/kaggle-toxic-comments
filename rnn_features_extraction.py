import os
import numpy as np

from utils import ToxicCommentsDataset

from keras.layers import Dense,Input,Bidirectional,Conv1D,GRU
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model

# laptop_data_dir = '/mnt/d/kaggle-toxic-comments/data'
filepath = "weights_base.best.hdf5"
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

embedding_matrix0 = np.random.randn(max_words, 300)
embedding_matrix1 = np.random.randn(max_words, 200)
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
model.load_weights(filepath=filepath)

del embedding_matrix0, embedding_matrix1

feature_extractor = Model(sequence_input, model.layers[-2].output)
feature_extractor.summary()
new_x_train = feature_extractor.predict(x_train, verbose=1)
new_x_test = feature_extractor.predict(x_test, verbose=1)

np.savetxt('new_x_train.out', new_x_train, delimiter=',')
np.savetxt('new_x_test.out', new_x_test, delimiter=',')