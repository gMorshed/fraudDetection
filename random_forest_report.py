import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # SVM classifier model
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA
from joblib import dump,load
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report


number_of_fraud_trans = 86
number_of_non_fraud_trans = 49914 
inputDataFrame = pd.read_csv('../train_data.csv')
inputDataFrame.dropna(inplace=True)
assert inputDataFrame.isnull().sum().sum() == 0
def f(name):
  if(name[0] == "C"):
    return 0
  elif(name[0]=="M"):
    return 1
  else:
    return NaN
  
  
inputDataFrame["isMerchantOrig"] = inputDataFrame["nameOrig"].apply(func=f)
inputDataFrame["isMerchantDest"] = inputDataFrame["nameDest"].apply(func=f)
inputDataFrame.drop(columns=["nameOrig","nameDest"], inplace=True)
new_inputDataFrame = pd.get_dummies(data=inputDataFrame["type"])
for item in new_inputDataFrame:
  inputDataFrame.insert(2, item,new_inputDataFrame[item]) 
inputDataFrame.drop(columns=["type","isFlaggedFraud"], inplace=True)
#samplying the non fradulent with all fradulent data
all_fraud = inputDataFrame.query('isFraud==1')
all_fraud= all_fraud.sample(n=number_of_fraud_trans, random_state=1,replace=True)
not_fraud = inputDataFrame.query('isFraud==0')
not_fraud= not_fraud.sample(n=number_of_non_fraud_trans, random_state=1,replace=True)
frames = [all_fraud,not_fraud]
inputDataFrame = pd.concat(frames)

#split test train
fraud_targets = pd.Series(inputDataFrame["isFraud"])
inputDataFrame.drop(columns=["isFraud"], inplace=True)
fraud_features = pd.DataFrame(inputDataFrame)
X_train, X_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)

clf = load('random_forest_w_all_feature.joblib') 
testScore = clf.score(X_test, y_test)
print("Random Forest + All feature Accuracy: ", testScore)
prediction_list = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,prediction_list).ravel()
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("True Positive: ", tp)
print("Best params: ",clf.best_params_)
print(classification_report(y_test, prediction_list, target_names=['fraud','non-fraud']))