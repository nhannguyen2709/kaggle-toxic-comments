import numpy as np
import os
import argparse

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import ToxicCommentsDataset, RocAucEvaluation
from bidirectionalgru import create_embeddings, BidirectionalGRU

parser = argparse.ArgumentParser(description='Bidirectional Gated Recurrent Unit network')
parser.add_argument('--data_dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
parser.add_argument('--output_dir', default='', type=str, metavar='PATH', help='path to model outputs (default: none)')
parser.add_argument('--embedding_dir', default='', type=str, metavar='PATH', help='path to the pre-trained word embeddings (default: None)')
parser.add_argument('--weights_filepath', default='', type=str, metavar='PATH', help='path to best weights saved (default: None)')
parser.add_argument('--train_csv_file', default='train.csv', type=str, metavar='PATH', help='train data filename')
parser.add_argument('--test_csv_file', default='test.csv', type=str, metavar='PATH', help='test data filename')
parser.add_argument('--new_x_train_file', default='new_x_train.out', type=str, metavar='PATH', help='processed train data filename')
parser.add_argument('--new_x_test_file', default='new_x_test.out', type=str, metavar='PATH', help='processed test data filename')
parser.add_argument('--max_words', default=100000, type=int, metavar='N', help='maximum size of the vocabulary')
parser.add_argument('--maxlen', default=150, type=int, metavar='N', help='maximum timestep of a sample')
parser.add_argument('--train_size', default=0.9, type=float, help='the proportion of the dataset to include in the training set')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate of the optimizer')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='number of samples in a batch')
parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of training epochs')
parser.add_argument('--num_classes', default=6, type=int, metavar='N', help='number of labels')

def main1():
    global arg
    arg = parser.parse_args()
    print(arg)
    
    embedding_filenames = sorted(os.listdir(arg.embedding_dir))
    embedding_sizes = [300, 200] # 
    # prepare the dataset
    toxic_comments_dataset = ToxicCommentsDataset(arg.data_dir,
                                                  arg.train_csv_file,
                                                  arg.test_csv_file)
    _, y_train, _ = toxic_comments_dataset.get_texts_and_train_labels()
    x_train, x_test, word_index = toxic_comments_dataset.tokenize_by_keras(max_words=arg.max_words, 
                                                                           maxlen=arg.maxlen)

    # obtain the pre-trained word embeddings
    list_embeddings_matrix = create_embeddings(embedding_dir=arg.embedding_dir,
                                               embedding_filenames=embedding_filenames,
                                               embedding_sizes=embedding_sizes,
                                               max_words=arg.max_words,
                                               word_index=word_index)
    
    # train-test split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, 
                                                                    train_size=arg.train_size, random_state=233)

    # create callbacks
    checkpoint = ModelCheckpoint(arg.weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
    roc_auc = RocAucEvaluation(validation_data=(x_validation, y_validation), interval=1)
    callbacks_list = [roc_auc,checkpoint, early]
 
    bidirectionalgru = BidirectionalGRU(input_shape=(arg.maxlen,), 
                                        list_embeddings_matrix=list_embeddings_matrix,
                                        weights_filepath=arg.weights_filepath,
                                        num_classes=arg.num_classes)
    bidirectionalgru.build_model()
    bidirectionalgru.train_last_layers(x_train=x_train, y_train=y_train,
                                       x_validation=x_validation, y_validation=y_validation,
                                       learning_rate=arg.learning_rate, batch_size=arg.batch_size,
                                       epochs=arg.epochs, callbacks_list=callbacks_list)
    
    # fine-tune the embedding layers
    bidirectionalgru.unfreeze_embeddings_and_train_all_layers(x_train=x_train, y_train=y_train,
                                                              x_validation=x_validation, y_validation=y_validation,
                                                              learning_rate=1e-5, batch_size=arg.batch_size,
                                                              epochs=arg.epochs, callbacks_list=callbacks_list)

def main2():
    global arg
    arg = parser.parse_args()
    print(arg)

    embedding_filenames = sorted(os.listdir(arg.embedding_dir))
    embedding_sizes = [300, 200]
    # prepare the dataset
    toxic_comments_dataset = ToxicCommentsDataset(arg.data_dir,
                                                  arg.train_csv_file,
                                                  arg.test_csv_file)
    x_train, x_test, word_index = toxic_comments_dataset.tokenize_by_keras(max_words=arg.max_words,
                                                                           maxlen=arg.maxlen)

    # obtain the pre-trained word embeddings
    list_embeddings_matrix = create_embeddings(embedding_dir=arg.embedding_dir,
                                               embedding_filenames=embedding_filenames,
                                               embedding_sizes=embedding_sizes,
                                               max_words=arg.max_words,
                                               word_index=word_index)

    # build the model
    bidirectionalgru = BidirectionalGRU(input_shape=(arg.maxlen,),
                                        list_embeddings_matrix=list_embeddings_matrix,
                                        weights_filepath=arg.weights_filepath,
                                        num_classes=arg.num_classes)
    bidirectionalgru.build_model()
    
    # extract features
    bidirectionalgru.reload_weights_from_checkpoint()
    new_x_train, new_x_test = bidirectionalgru.extract_features(x_train=x_train, x_test=x_test)
    np.savetxt(os.path.join(arg.output_dir, arg.new_x_train_file), 
               new_x_train, delimiter=',')
    np.savetxt(os.path.join(arg.output_dir, arg.new_x_test_file),
               new_x_test, delimiter=',')

if __name__=='__main__':
    main1()
    main2()