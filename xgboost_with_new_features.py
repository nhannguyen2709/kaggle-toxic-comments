import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn_model_pipelines import save_outputs

from utils import ToxicCommentsDataset

desktop_data_dir = '/home/nhan/Downloads/toxic_comments'
toxic_comments_dataset = ToxicCommentsDataset(desktop_data_dir,
                                              'train.csv',
                                              'test.csv')

_, y_train, _ = toxic_comments_dataset.get_texts_and_train_labels()
new_x_train = np.loadtxt('new_x_train.out', delimiter=',')
new_x_test = np.loadtxt('new_x_test.out', delimiter=',')
num_train_samples = len(new_x_train)
num_test_samples = len(new_x_train)
num_classes = 6
num_iter = 100
scoring = 'roc_auc'
num_jobs = 4
kfold = 3
model_outputs = np.zeros((num_train_samples + num_test_samples, num_classes))
xgbclassifier = XGBClassifier()
for i in range(num_classes): # iterate through labels
    param_grid = {'n_estimators': np.array([5, 10, 15, 20, 25]),
                  'max_depth': np.array([5,10,15,20,25]),
                  'subsample': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                  'colsample_bytree': np.array([0.5,0.6,0.7,0.8,0.9,1.0]),
                  'learning_rate': np.array([0.01,0.05,0.10,0.20,0.30,0.40]),
                  'gamma': np.array([0.00,0.05,0.10,0.15,0.20]),
                  'scale_pos_weight': np.array([30,40,50,300,400,500,600,700])}
    randomized = RandomizedSearchCV(xgbclassifier, param_distributions=param_grid,
                                    n_iter=num_iter, scoring=scoring,
                                    n_jobs=num_jobs, cv=kfold)
    randomized.fit(new_x_train, y_train[:, i])
    y_train_pred = randomized.predict_proba(new_x_train)
    y_test_pred = randomized.predict_proba(new_x_test)
    best_params_dict = randomized.best_params_
    print('Best hyperparameters set found for label {}: {}'.format(i, best_params_dict))
    model_outputs[:num_train_samples, i] = y_train_pred[:, 1]
    model_outputs[num_train_samples:, i] = y_test_pred[:, 1]

# save outputs
save_outputs(model_outputs, desktop_data_dir, 'train.csv', 'xgb_outputs.csv')