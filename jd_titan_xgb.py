
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')

# Uploading the data into a pandas DataFrame
titan_data = pd.read_csv('/Users/HomeFolder/Python1/Machine_learning_course/titanic/train.csv')

# Here I set up the features (predictors) and dropped Survived because it is the target variable
# I dropped Name and Ticket, because by doing so I was able to achieve lower Mean Absolute Errors with the models implemented
titan_preds = titan_data.drop(['Survived','Name','Ticket'], axis=1)

# Here I convert the categorical features to one hot encoding
# one hot encoding translates the values of the categorical features into binary form
# in which each distinct value (red, green, blue) is given a column 
# and if that value appears, then it's cell/row value is set to 1, and the other distinct values have cells of value 0
# One hot encoding is used so that machine learning models can interpret categorical data
oh_titan_preds = pd.get_dummies(titan_preds)

# Here I create an Series (dataframe) of the target variable Survived which is = 1 if
# the passenger survived and 0 if they did not
titan_targ = titan_data.Survived

# Here I split the training data into test and training data 
# I didn't distinguish how much of the data should be test and how much should be split
# which is something I should do if I want to improve this code/model
train_X, val_X, train_y, val_y = train_test_split(oh_titan_preds, titan_targ, random_state = 0)


def cross_val_score_f(oh_titan_preds, titan_targ, method):
    """ This function creates a pipeline that uses imputation (for null values) and the XGBClassifier.
    I use Cross Validation to check my model's accuracy.
    """
    my_pipeline = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators=113, learning_rate=0.05))
    my_pipeline.fit(oh_titan_preds, titan_targ)
    prediction = my_pipeline.predict(val_X)
    scores = cross_val_score(my_pipeline, oh_titan_preds, titan_targ,
    scoring='neg_mean_absolute_error',cv=3)
    print(f"Your MAE's from cross validation for the {method} are: ")
    print(scores*-1)


def score_boost(train_X, val_X, train_y, val_y):
    """ I made this function to compare the effect of imputation on the model."""
    model = XGBClassifier(n_estimators=113, learning_rate=0.05)
    model.fit(train_X, train_y,
              eval_set=[(val_X, val_y)], verbose=False)
    preds = model.predict(val_X)
    print("MAE for XGBoost:")
    return mean_absolute_error(val_y, preds)

print(score_boost(train_X, val_X, train_y, val_y))

cross_val_score_f(oh_titan_preds, titan_targ, "XGB Pipeline")

# creating a DataFrame from the test.csv file 
test_data = pd.read_csv('/Users/HomeFolder/Python1/Machine_learning_course/titanic/test.csv')




# Here I define which rows I would like to be considered as features in my final model
cols_to_use = ['PassengerId','Age','Pclass','SibSp','Parch','Fare','Cabin','Embarked','Sex']
#Setting up the features (from the DataFrame)
test_X_not_one_hot = test_data[cols_to_use]
test_X = pd.get_dummies(test_X_not_one_hot)
# I also need to align the test and training data... (b/c differ # of rows)
final_train, final_test = oh_titan_preds.align(
test_X, join='right', axis =1)


# I'm using imputation b/c age, cabin, and Embarked have missing values.
test_model = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators=113, learning_rate=0.05))
test_model.fit(final_train, titan_targ)

test_preds = test_model.predict(final_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                      'Survived': test_preds})

output.to_csv('submission.csv', index=False)
