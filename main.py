import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

number_of_fraud_trans = 86
number_of_non_fraud_trans = 49914

inputDataFrame = pd.read_csv('../data.csv')
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
all_fraud= all_fraud.sample(n=number_of_fraud_trans, random_state=1)
not_fraud = inputDataFrame.query('isFraud==0')
not_fraud= not_fraud.sample(n=number_of_non_fraud_trans, random_state=1)
frames = [all_fraud,not_fraud]
inputDataFrame = pd.concat(frames)

#split test train 
fraud_targets = pd.Series(inputDataFrame["isFraud"])
inputDataFrame.drop(columns=["isFraud"], inplace=True)
fraud_features = pd.DataFrame(inputDataFrame)
X_train, X_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)
#feature selection
k=5
mi_transformer = SelectKBest(mutual_info_regression,k=k).fit(X_train, y_train)
mi_X_train,mi_X_test = mi_transformer.transform(X_train), mi_transformer.transform(X_test)

for feature, importance in zip(fraud_features.columns, mi_transformer.scores_):
    print(f"The MI score for {feature} is {importance}")

#setting uyp gridsearchCV
n_estimators = [10, 30]
max_depth = [2,5, 8]
min_samples_split = [2,3]
min_samples_leaf = [1, 2, 5] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
forest = RandomForestClassifier(n_estimators=100,random_state=0)

#Random forest with all features
clf_w_allFeatures = GridSearchCV(forest, hyperF, cv=5)
clf_w_allFeatures.fit(X_train,y_train)
print("RandomForestClassifier score without any feature selection is :",clf_w_allFeatures.score(X_test,y_test))
print("Best parameter with all features: ",clf_w_allFeatures.best_params_)
predictions = clf_w_allFeatures.predict(X_test)
conf_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix with all features ",conf_mat)

#Random forest with selected features
clf_w_selected_features = GridSearchCV(forest, hyperF, cv=5)
clf_w_selected_features.fit(mi_X_train,y_train)
print("RandomForestClassifier score with feature selection is :",clf_w_selected_features.score(mi_X_test,y_test))
print("Best parameter with selected features: ",clf_w_selected_features.best_params_)
predictions = clf_w_selected_features.predict(mi_X_test)
conf_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix with selected features ",conf_mat)