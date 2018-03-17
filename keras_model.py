import numpy as np
import os

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import ToxicCommentsDataset, RocAucEvaluation
from bidirectionalgru import create_embeddings, BidirectionalGRU

laptop_data_dir = '/mnt/d/kaggle-toxic-comments/data'
desktop_data_dir = '/home/nhan/Downloads/toxic_comments'
embedding_dir = '/home/nhan/Downloads/word_embeddings/'
filepath = "weights_base.best.hdf5"

toxic_comments_dataset = ToxicCommentsDataset(laptop_data_dir,
                                              'train.csv',
                                              'test.csv')
max_words = 100000
maxlen = 150
_, y_train, _ = toxic_comments_dataset.get_texts_and_train_labels()
x_train, x_test, word_index = toxic_comments_dataset.tokenize_by_keras(max_words=max_words, maxlen=maxlen)

# create_embeddings()
list_embeddings_matrix = [np.random.randn(max_words, 30), np.random.randn(max_words, 10)]

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.9, random_state=233)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
roc_auc = RocAucEvaluation(validation_data=(x_validation, y_validation), interval = 1)
callbacks_list = [roc_auc,checkpoint, early]
 
bidirectionalgru = BidirectionalGRU(input_shape=(maxlen,), 
                                    list_embeddings_matrix=list_embeddings_matrix,
                                    weights_filepath=filepath,
                                    num_classes=6)
bidirectionalgru.build_model()
np.savetxt('new_x_train.out', new_x_train, delimiter=',')
np.savetxt('new_x_test.out', new_x_test, delimiter=',')
