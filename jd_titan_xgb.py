# Terminal Commands:

# cd ~/Python1/Machine_learning_course/titanic

# python3.6 jd_titan_xgb.py

# I'm not going to do cross-validaiton b/c that requires
# using GridSearchCV and I still don't understand those
# results... so for now I'll just use the train_test_split
# XGB and MAE.

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib as plt

titan_data = pd.read_csv('/Users/HomeFolder/Python1/Machine_learning_course/titanic/train.csv')


titan_preds = titan_data.drop(['Survived','Name','Ticket'], axis=1)
oh_titan_preds = pd.get_dummies(titan_preds)
titan_targ = titan_data.Survived

train_X, val_X, train_y, val_y = train_test_split(oh_titan_preds, titan_targ, random_state = 0)

def cross_val_score_f(oh_titan_preds, titan_targ, method):
    my_pipeline = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators=113, learning_rate=0.05))
    my_pipeline.fit(oh_titan_preds, titan_targ)
    prediction = my_pipeline.predict(val_X)
    scores = cross_val_score(my_pipeline, oh_titan_preds, titan_targ,
    scoring='neg_mean_absolute_error',cv=3)
    print(f"Your MAE's from cross validation for the {method} are: ")
    # OK I don't really understand what the MAE is in reference to...
    # Since we are trying to predict 0 or 1... I guess an MAE of 0.2 isn't that bad
    # whereas an MAE of 0.5 would be pretty bad...
    print(scores*-1)


def score_boost(train_X, val_X, train_y, val_y):
    model = XGBClassifier(n_estimators=113, learning_rate=0.05)
    model.fit(train_X, train_y,
              eval_set=[(val_X, val_y)], verbose=False)
    preds = model.predict(val_X)
    print("MAE for XGBoost:")
    return mean_absolute_error(val_y, preds)

print(score_boost(train_X, val_X, train_y, val_y))

cross_val_score_f(oh_titan_preds, titan_targ, "XGB Pipeline")

test_data = pd.read_csv('/Users/HomeFolder/Python1/Machine_learning_course/titanic/test.csv')


# I need to convert the test_data into one-hot encoded values!
# I also need to align them... (b/c differ # of rows)

cols_to_use = ['PassengerId','Age','Pclass','SibSp','Parch','Fare','Cabin','Embarked','Sex']

test_X_not_one_hot = test_data[cols_to_use]
test_X = pd.get_dummies(test_X_not_one_hot)

final_train, final_test = oh_titan_preds.align(
test_X, join='right', axis =1)

# make_pipeline(SimpleImputer(),
# I'm using imputation b/c age, cabin, and Embarked have missing values.
test_model = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators=113, learning_rate=0.05))
test_model.fit(final_train, titan_targ)

test_preds = test_model.predict(final_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                      'Survived': test_preds})

output.to_csv('submission.csv', index=False)
