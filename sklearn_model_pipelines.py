import numpy as np
import pandas as pd
import os
import argparse 

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from utils import ToxicCommentsDataset


parser = argparse.ArgumentParser(description='Scikit-learn text vectorizer and classifier pipelines')
parser.add_argument('--data_dir', default='', type=str, metavar='PATH', help='path to data (default: none)')
parser.add_argument('--train_csv_file', default='train.csv', metavar='PATH', help='train data filename')
parser.add_argument('--test_csv_file', default='test.csv', metavar='PATH', help='test data filename')
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
    train_texts, y_train, test_texts = toxic_comments_dataset.get_texts_and_train_labels()
    # run cross-validation to tune the hyperparameters
    trial = ScikitLearnPipeline(train_texts, test_texts, y_train,
                                arg.num_iter, arg.scoring, arg.num_jobs, 
                                arg.kfold, arg.num_classes)
    xgb_outputs = trial.tf_idf_xgboost_pipeline(silent=False)
    save_outputs(xgb_outputs, arg.data_dir, arg.train_csv_file, 'xgb_outputs.csv') # save the outputs
    rf_outputs = trial.tf_idf_random_forest_pipeline(verbose=1)
    save_outputs(rf_outputs, arg.data_dir, arg.train_csv_file, 'rf_outputs.csv') # save the outputs

class ScikitLearnPipeline:
    """Build pipelines of text transformer and classifiers using RandomizedCV() to choose the best hyperparameters."""
    def __init__(self, train_texts, test_texts, y_train,
                 num_iter, scoring, num_jobs, kfold, num_classes):
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.y_train = y_train
        self.num_iter = num_iter
        self.scoring = scoring
        self.num_jobs = num_jobs
        self.kfold = kfold
        self.num_classes = num_classes
    
    def tf_idf_random_forest_pipeline(self, verbose):
        """Tf-idf and random forest pipeline.
        
        # Args
            verbose (int, default=0): controls the verbosity of the tree building process
                    
        # Returns
            A Numpy array of predicted probabilities on both train and test data.
        """
        
        num_train_samples = len(self.train_texts)
        num_test_samples = len(self.test_texts)
        model_outputs = np.zeros((num_train_samples + num_test_samples, self.num_classes))
        for i in range(self.num_classes): # iterate through labels
            text_pipe = make_pipeline(TfidfVectorizer(), RandomForestClassifier(verbose=verbose))
            param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)], 
                          'randomforestclassifier__n_estimators': np.arange(10, 25 + 1),
                          'randomforestclassifier__min_samples_split': np.arange(2, 20 + 1),
                          'randomforestclassifier__max_features': ['sqrt', 'log2']}
            randomized = RandomizedSearchCV(text_pipe, param_distributions=param_grid, 
                                            n_iter=self.num_iter, scoring=self.scoring,
                                            n_jobs=self.num_jobs, cv=self.kfold)
            randomized.fit(self.train_texts, self.y_train[:, i])
            y_train_pred = randomized.predict_proba(self.train_texts) 
            y_test_pred = randomized.predict_proba(self.test_texts)
            best_params_dict = randomized.best_params_
            print('Best hyperparameters set found for label {}: {}'.format(i, best_params_dict))
            model_outputs[:num_train_samples, i] = y_train_pred[:, 1]
            model_outputs[num_train_samples:, i] = y_test_pred[:, 1]
        return model_outputs
    
    def tf_idf_xgboost_pipeline(self, silent):
        """Tf-idf and xgboost pipeline.
        
        # Args
            silent (boolean): whether to print messages while running boosting

        # Returns
            A Numpy array of predicted probabilities on both train and test data.
        """
        
        num_train_samples = len(self.train_texts)
        num_test_samples = len(self.test_texts)
        model_outputs = np.zeros((num_train_samples + num_test_samples, self.num_classes))
        for i in range(self.num_classes): # iterate through labels
            text_pipe = make_pipeline(TfidfVectorizer(), XGBClassifier(silent=silent))
            param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)], 
                          'xgbclassifier__n_estimators': np.array([5, 10, 15, 20, 25]),
                          'xgbclassifier__max_depth': np.array([5,10,15,20,25]),
                          'xgbclassifier__subsample': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                          'xgbclassifier__colsample_bytree': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                          'xgbclassifier__learning_rate': np.array([0.01,0.05,0.10,0.20,0.30,0.40]),
                          'xgbclassifier__gamma': np.array([0.00,0.05,0.10,0.15,0.20]),
                          'xgbclassifier__scale_pos_weight': np.array([30,40,50,300,400,500,600,700])}
            randomized = RandomizedSearchCV(text_pipe, param_distributions=param_grid, 
                                            n_iter=self.num_iter, scoring=self.scoring,
                                            n_jobs=self.num_jobs, cv=self.kfold)
            randomized.fit(self.train_texts, self.y_train[:, i])
            y_train_pred = randomized.predict_proba(self.train_texts) 
            y_test_pred = randomized.predict_proba(self.test_texts)
            best_params_dict = randomized.best_params_
            print('Best hyperparameters set found for label {}: {}'.format(i, best_params_dict))
            model_outputs[:num_train_samples, i] = y_train_pred[:, 1]
            model_outputs[num_train_samples:, i] = y_test_pred[:, 1]
        return model_outputs

def save_outputs(model_outputs, data_dir, train_csv_file, output_csv_file):
    classes = pd.read_csv(os.path.join(data_dir, train_csv_file)).columns[2:]
    model_outputs_df = pd.DataFrame(model_outputs, columns=classes)
    model_outputs_df.to_csv(output_csv_file, index=False)
    print('Finished exporting the model outputs to a csv file.')    

if __name__=='__main__':
    main()