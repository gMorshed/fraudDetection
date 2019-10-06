import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # SVM classifier model


inputDataFrame = pd.read_csv('data.csv')
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
not_fraud = inputDataFrame.query('isFraud==0')
not_fraud= not_fraud.sample(n=40000, random_state=1)
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


#SVM model
hyperparams = {
    "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4],
    "random_state": [0]
}
svc = SVC(gamma='auto')
clf = GridSearchCV(svc, hyperparams, scoring='accuracy')
clf.fit(mi_X_train, y_train)
testScore = clf.score(mi_X_test, y_test)
print("SVM Accuracy: ", testScore)
optional = False
if(optional):
  C = 1.0  # SVM regularization parameter
  models = (sk.svm.SVC(kernel='linear', C=C),
    sk.svm.LinearSVC(C=C),
    sk.svm.SVC(kernel='rbf', gamma=0.7, C=C),
    sk.svm.SVC(kernel='poly', degree=3, C=C))
  models = (clf.fit(X, y) for clf in models)











