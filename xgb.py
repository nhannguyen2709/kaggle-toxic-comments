import numpy as np
import pandas as pd
import os 

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

from dataset import ToxicCommentsDataset

toxic_comments_dataset = ToxicCommentsDataset('/home/nhan/Downloads/toxic_comments',
                                              'train.csv', 'test.csv')
train_texts, y_train, test_texts = toxic_comments_dataset.get_texts_and_train_labels()
classes = pd.read_csv(os.path.join('/home/nhan/Downloads/toxic_comments', 'train.csv')).columns[2:]

def tf_idf_xgb_pipeline(train_texts, test_texts, y_train, 
                        num_classes, num_jobs, num_iter, scoring_method='roc_auc'):
    num_train_samples = len(train_texts)
    num_test_samples = len(test_texts)
    model_outputs = np.zeros((num_train_samples + num_test_samples, num_classes))
    for i in range(num_classes):
        text_pipe = make_pipeline(TfidfVectorizer(), xgb.XGBClassifier(silent=False))
        param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)], 
                      'xgbclassifier__max_depth': np.array([5,10,15,20,25]),
                      'xgbclassifier__subsample': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                      'xgbclassifier__colsample_bytree': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                      'xgbclassifier__learning_rate': np.array([0.01,0.05,0.10,0.20,0.30,0.40]),
                      'xgbclassifier__gamma': np.array([0.00,0.05,0.10,0.15,0.20]),
                      'xgbclassifier__scale_pos_weight': np.array([30,40,50,300,400,500,600,700])}
        randomized = RandomizedSearchCV(text_pipe, param_distributions=param_grid, 
                                        cv=3, scoring=scoring_method)
        randomized.fit(train_texts, y_train[:, i])
        y_train_pred = randomized.predict_proba(train_texts)
        y_test_pred = randomized.predict_proba(test_texts)
        model_outputs[:num_train_samples, i] = y_train_pred
        model_outputs[num_train_samples:, i] = y_test_pred
    return model_outputs


xgb_outputs = tf_idf_xgb_pipeline(train_texts, test_texts, y_train, 6, 4, 50)
xgb_outputs = pd.DataFrame(xgb_outputs)
xgb_outputs.columns = classes
xgb_outputs.to_csv('xgb_outputs.csv', index=False)