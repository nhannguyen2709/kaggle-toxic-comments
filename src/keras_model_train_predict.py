import numpy as np
import os
import argparse

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils import save_outputs, ToxicCommentsDataset, RocAucEvaluation
from bidirectionalgru import create_embeddings, BidirectionalGRU

parser = argparse.ArgumentParser(
    description='Bidirectional Gated Recurrent Unit network')
parser.add_argument('--data_dir', default='', type=str,
                    metavar='PATH', help='path to data (default: none)')
parser.add_argument('--output_dir', default='', type=str,
                    metavar='PATH', help='path to model outputs (default: none)')
parser.add_argument('--embedding_dir', default='', type=str, metavar='PATH',
                    help='path to the pre-trained word embeddings (default: None)')
parser.add_argument('--reload', default=True, type=bool, 
                    help='whether to reload model weights from last checkpoint')
parser.add_argument('--weights_filepath', default='', type=str,
                    metavar='PATH', help='path to best weights saved (default: None)')
parser.add_argument('--modelcheckpoint_filepath', default='weights.best.hdf5',
                    type=str, metavar='PATH', help='path to checkpoint model weights')
parser.add_argument('--train_csv_file', default='train.csv',
                    type=str, metavar='PATH', help='train data filename')
parser.add_argument('--test_csv_file', default='test.csv',
                    type=str, metavar='PATH', help='test data filename')
parser.add_argument('--new_x_train_file', default='new_x_train.out',
                    type=str, metavar='PATH', help='processed train data filename')
parser.add_argument('--new_x_test_file', default='new_x_test.out',
                    type=str, metavar='PATH', help='processed test data filename')
parser.add_argument('--spatial_dropout1d_rate', default=0.2,
                    type=float, help='dropout rate of the SpatialDropout1D layer')
parser.add_argument('--hidden_dim', default=256, type=int,
                    metavar='N', help='number of hidden units of the GRU layer')
parser.add_argument('--dropout_rate', default=0.2, type=float,
                    help='dropout rate of the GRU layer')
parser.add_argument('--recurrent_dropout_rate', default=0.2,
                    type=float, help='recurrent dropout rate of the GRU layer')
parser.add_argument('--num_conv1d_filters', default=128, type=int,
                    metavar='N', help='number of filters of the Conv1D layer')
parser.add_argument('--max_words', default=100000, type=int,
                    metavar='N', help='maximum size of the vocabulary')
parser.add_argument('--maxlen', default=150, type=int,
                    metavar='N', help='maximum timestep of a sample')
parser.add_argument('--train_size', default=0.9, type=float,
                    help='the proportion of the dataset to include in the training set')
parser.add_argument('--learning_rate', default=1e-3,
                    type=float, help='learning rate of the optimizer')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N', help='number of samples in a batch')
parser.add_argument('--epochs', default=15, type=int,
                    metavar='N', help='number of training epochs')
parser.add_argument('--num_train_test_splits', default=10, type=int,
                    metavar='N', help='number of times to split the training set')
parser.add_argument('--num_classes', default=6, type=int,
                    metavar='N', help='number of labels')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    embedding_filenames = sorted(os.listdir(arg.embedding_dir))
    embedding_sizes = [300, 100, 200, 50, 300]
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

    checkpoint = ModelCheckpoint(arg.modelcheckpoint_filepath,
                                 monitor='val_acc', verbose=1, 
                                 save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)

    bidirectionalgru = BidirectionalGRU(input_shape=(arg.maxlen,),
                                        list_embeddings_matrix=list_embeddings_matrix,
                                        num_classes=arg.num_classes,
                                        weights_filepath=arg.weights_filepath,
                                        spatial_dropout1d_rate=arg.spatial_dropout1d_rate,
                                        hidden_dim=arg.hidden_dim,
                                        dropout_rate=arg.dropout_rate,
                                        recurrent_dropout_rate=arg.recurrent_dropout_rate,
                                        num_conv1d_filters=arg.num_conv1d_filters)
    bidirectionalgru.build_model()
    if arg.reload:
        bidirectionalgru.reload_weights_from_checkpoint()

    for i in range(arg.num_train_test_splits):
        print('Start training the model on split {}'.format(i + 1))
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,
                                                                        train_size=arg.train_size, random_state=i)
        roc_auc = RocAucEvaluation(validation_data=(
            x_validation, y_validation), interval=1)
        callbacks_list = [roc_auc, checkpoint, early, reduce_lr]
        bidirectionalgru.train_last_layers(x_train=x_train, y_train=y_train,
                                           x_validation=x_validation, y_validation=y_validation,
                                           learning_rate=arg.learning_rate, batch_size=arg.batch_size,
                                           epochs=arg.epochs, callbacks_list=callbacks_list)
        bidirectionalgru.unfreeze_embeddings_and_train_all_layers(x_train=x_train, y_train=y_train,
                                                                  x_validation=x_validation, y_validation=y_validation,
                                                                  learning_rate=1e-5, batch_size=arg.batch_size,
                                                                  epochs=arg.epochs, callbacks_list=callbacks_list)
        bigru_preds = bidirectionalgru.predict_on_test_data(x_test=x_test)
        save_outputs(bigru_preds, arg.data_dir, arg.output_dir,
                     arg.train_csv_file, 'bigru_outputs'+str(i+1)+'.csv')

if __name__=='__main__':
    main()