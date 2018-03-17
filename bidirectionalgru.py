import os
import numpy as np

from keras.layers import Dense, Input, Bidirectional, Conv1D, GRU
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, SpatialDropout1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model 
from keras.optimizers import Adam


def create_embeddings(embedding_dir, embedding_filenames, 
                      embedding_sizes, max_words, word_index):
    list_embeddings_index = []
    for filename in embedding_filenames:
        if filename.endswith('.txt'):          
            embedding_index = {}
            with open(os.path.join(embedding_dir, filename), encoding='utf8') as f:
                for line in f:
                    values = line.rstrip().rsplit(' ')
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_index[word] = coefs
            f.close()
            list_embeddings_index.append(embedding_index)
        else:
            embedding_index = {}
            with open(os.path.join(embedding_dir, filename), mode='rb') as f:
                for line in f:
                    values = line.rstrip().rsplit(' ')
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_index[word] = coefs
            f.close()
            list_embeddings_index.append(embedding_index)

    list_embeddings_matrix = []
    for ix, embedding_size in enumerate(embedding_sizes):
        num_words = min(max_words, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, embedding_size))
        for word, i in word_index.items():
            if i >= max_words:
                continue
            embedding_vector = list_embeddings_index[ix].get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        list_embeddings_matrix.append(embedding_matrix)
    
    del list_embeddings_index

    return list_embeddings_matrix


class BidirectionalGRU:
    def __init__(self, 
                 input_shape, 
                 list_embeddings_matrix, 
                 weights_filepath, 
                 num_classes):
        self.input_shape = input_shape
        self.list_embeddings_matrix = list_embeddings_matrix
        self.num_classes = num_classes
        self.weights_filepath = weights_filepath

    def build_model(self, verbose=True):
        list_x_embedded = []
        self.sequence_input = Input(shape=self.input_shape)
        
        # load the embeddings matrix into Embedding() layer and concatenate them
        for i in range(len(self.list_embeddings_matrix)):
            embedding_matrix = self.list_embeddings_matrix[i]
            max_words = embedding_matrix.shape[0]
            embedding_size = embedding_matrix.shape[1]
            x_embedded = Embedding(max_words, embedding_size, 
                                   weights=[embedding_matrix], trainable=False)(self.sequence_input)
            list_x_embedded.append(x_embedded)
        x = Concatenate(list_x_embedded)

        x = SpatialDropout1D(0.2)
        x = Bidirectional(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
        x = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate([avg_pool, max_pool])
        outputs = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model(self.sequence_input, outputs)
        if verbose:
            self.model.summary()

    def train_last_layers(self, 
                          x_train, y_train,
                          x_validation, y_validation,
                          learning_rate, batch_size, 
                          epochs, callbacks_list):
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate),
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, 
                       batch_size=batch_size, epochs=epochs,
                       validation_data=(x_validation, y_validation),
                       callbacks=callbacks_list, verbose=1)

    def unfreeze_embeddings_and_train_all_layers(self,
                                                 x_train, y_train,
                                                 x_validation, y_validation,
                                                 learning_rate, batch_size,
                                                 epochs, callbacks_list):
        for layer in self.model.layers[1:len(self.list_embeddings_matrix) + 1]:
            layer.trainable = True     
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate),
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       validation_data=(x_validation, y_validation),
                       callbacks=callbacks_list, verbose=1)

    def reload_weights_from_checkpoint(self):
        self.model.load(filepath=self.weights_filepath)

    def extract_features(self, x_train, x_test, verbose=True): 
        feature_extractor = Model(self.sequence_input, self.model.layers[-2].output)
        if verbose:
            feature_extractor.summary()
        new_x_train = feature_extractor.predict(x_train, verbose=1)
        new_x_test = feature_extractor.predict(x_test, verbose=1)
        return new_x_train, new_x_test
