import numpy as np
import pandas as pd
import os
import argparse 

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from utils import ToxicCommentsDataset, save_outputs


parser = argparse.ArgumentParser(description='Scikit-learn text vectorizer and classifier pipelines')
parser.add_argument('--data_dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
parser.add_argument('--output_dir', default='', type=str, metavar='PATH', help='path to model outputs (default: none)')
parser.add_argument('--train_csv_file', default='train.csv', type=str, metavar='PATH', help='train data filename')
parser.add_argument('--test_csv_file', default='test.csv', type=str, metavar='PATH', help='test data filename')
parser.add_argument('--new_x_train_file', default='new_x_train.out', type=str, metavar='PATH', help='processed train data filename')
parser.add_argument('--new_x_test_file', default='new_x_test.out', type=str, metavar='PATH', help='processed test data filename')
parser.add_argument('--num_iter', default=50, type=int, metavar='N', help='number of hyperparameter settings that are sampled')
parser.add_argument('--scoring', default='roc_auc', type=str, metavar='SCORING', help='metric to evaluate the predictions on the test set')
parser.add_argument('--num_jobs', default=8, type=int, metavar='N', help='number of jobs to run in parallel')
parser.add_argument('--kfold', default=3, type=int, metavar='N', help='number of folds in K-fold cross-validation')
parser.add_argument('--num_classes', default=6, type=int, metavar='N', help='number of labels')

def main():
    global arg
    arg = parser.parse_args()
    print(arg)
    
    # prepare the dataset
    toxic_comments_dataset = ToxicCommentsDataset(arg.data_dir, arg.train_csv_file, arg.test_csv_file)
    _, y_train, _ = toxic_comments_dataset.get_texts_and_train_labels()
    x_train = np.loadtxt(os.path.join(arg.output_dir, arg.new_x_train_file), delimiter=',')
    x_test = np.loadtxt(os.path.join(arg.output_dir, arg.new_x_test_file), delimiter=',')
    # run cross-validation to tune the hyperparameters
    trial = ScikitLearnClassifiers(x_train, x_test, y_train,
                                   arg.num_iter, arg.scoring, arg.num_jobs, 
                                   arg.kfold, arg.num_classes)
    xgb_outputs = trial.xgbclassifier(silent=False)
    save_outputs(xgb_outputs, arg.data_dir, arg.output_dir, arg.train_csv_file, 'xgb_outputs.csv') # save the outputs
    etc_outputs = trial.extra_trees_classifier(verbose=1)
    save_outputs(etc_outputs, arg.data_dir, arg.output_dir, arg.train_csv_file, 'etc_outputs.csv') # save the outputs

class ScikitLearnClassifiers:
    """Build pipelines of text transformer and classifiers using RandomizedCV() to choose the best hyperparameters."""
    def __init__(self, x_train, x_test, y_train,
                 num_iter, scoring, num_jobs, kfold, num_classes):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.num_iter = num_iter
        self.scoring = scoring
        self.num_jobs = num_jobs
        self.kfold = kfold
        self.num_classes = num_classes
    
    def extra_trees_classifier(self, verbose):
        """Extra trees classifier.
        
        # Args
            verbose (int, default=0): controls the verbosity of the tree building process
                    
        # Returns
            A Numpy array of predicted probabilities on both train and test data.
        """
        
        num_train_samples = len(self.x_train)
        num_test_samples = len(self.x_test)
        model_outputs = np.zeros((num_train_samples + num_test_samples, self.num_classes))
        param_grid = {'n_estimators': np.arange(100, 1000, 50),
                      'max_depth': np.array([1, 5, 10, 15, 20, 25, 30]),
                      'min_samples_leaf': np.array([1, 2, 4, 6, 8, 10]),
                      'min_samples_split': np.arange(2, 20 + 1),
                      'max_features': ['sqrt', 'log2']}
        for i in range(self.num_classes):
            etc = ExtraTreesClassifier(verbose=verbose)              
            randomized = RandomizedSearchCV(etc, param_distributions=param_grid, 
                                            n_iter=self.num_iter, scoring=self.scoring,
                                            n_jobs=self.num_jobs, cv=self.kfold)
            randomized.fit(self.x_train, self.y_train[:, i])
            y_train_pred = randomized.predict_proba(self.x_train) 
            y_test_pred = randomized.predict_proba(self.x_test)
            best_params_dict = randomized.best_params_
            best_score = randomized.best_score_
            print('Best hyperparameters set found for label {}: {}, mean cross-validated AUC of the best estimator: {}'
                  .format(i, best_params_dict, best_score))
            model_outputs[:num_train_samples, i] = y_train_pred[:, 1]
            model_outputs[num_train_samples:, i] = y_test_pred[:, 1]
        return model_outputs
    
    def xgbclassifier(self, silent):
        """Xgboost classifier.
        
        # Args
            silent (boolean): whether to print messages while running boosting

        # Returns
            A Numpy array of predicted probabilities on both train and test data.
        """
        
        num_train_samples = len(self.x_train)
        num_test_samples = len(self.x_test)
        model_outputs = np.zeros((num_train_samples + num_test_samples, self.num_classes))
        for i in range(self.num_classes): # iterate through labels
            xgb = XGBClassifier(silent=silent)
            param_grid = {'n_estimators': np.array([5, 10, 15, 20, 25]),
                          'max_depth': np.array([5,10,15,20,25]),
                          'subsample': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                          'colsample_bytree': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                          'learning_rate': np.array([0.01,0.05,0.10,0.20,0.30,0.40]),
                          'gamma': np.array([0.00,0.05,0.10,0.15,0.20]),
                          'scale_pos_weight': np.array([30,40,50,300,400,500,600,700])}
            randomized = RandomizedSearchCV(xgb, param_distributions=param_grid, 
                                            n_iter=self.num_iter, scoring=self.scoring,
                                            n_jobs=self.num_jobs, cv=self.kfold)
            randomized.fit(self.x_train, self.y_train[:, i])
            y_train_pred = randomized.predict_proba(self.x_train) 
            y_test_pred = randomized.predict_proba(self.x_test)
            best_params_dict = randomized.best_params_
            best_score = randomized.best_score_
            print('Best hyperparameters set found for label {}: {}, mean cross-validated AUC of the best estimator: {}'
                  .format(i, best_params_dict, best_score))
            model_outputs[:num_train_samples, i] = y_train_pred[:, 1]
            model_outputs[num_train_samples:, i] = y_test_pred[:, 1]
        return model_outputs

if __name__=='__main__':
    main()
