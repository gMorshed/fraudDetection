import pandas as pd
import numpy as np
import io
import copy
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE

number_of_fraud_trans = 86
number_of_non_fraud_trans = 49914

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
all_fraud= all_fraud.sample(n=number_of_fraud_trans, random_state=1)
not_fraud = inputDataFrame.query('isFraud==0')
not_fraud= not_fraud.sample(n=number_of_non_fraud_trans, random_state=1)
frames = [all_fraud,not_fraud]
inputDataFrame = pd.concat(frames)

#split test train
fraud_targets = pd.Series(inputDataFrame["isFraud"])
inputDataFrame_seq_cov = copy.deepcopy(inputDataFrame)
inputDataFrame.drop(columns=["isFraud"], inplace=True)
fraud_features = pd.DataFrame(inputDataFrame)
fraud_features_seq_cov = pd.DataFrame(inputDataFrame_seq_cov)
X_train, X_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)
#feature selection
k=5
mi_transformer = SelectKBest(mutual_info_regression,k=k).fit(X_train, y_train)
mi_X_train,mi_X_test = mi_transformer.transform(X_train), mi_transformer.transform(X_test)

for feature, importance in zip(fraud_features.columns, mi_transformer.scores_):
    print(f"The MI score for {feature} is {importance}")

#Sequential covering
X_train_seq_cov, X_test_seq_cov, y_train_seq_cov, y_test_seq_cov = train_test_split(fraud_features_seq_cov, fraud_targets, test_size=0.2, random_state=0)
class_feat = "isFraud"
clf = lw.RIPPER()
clf.fit(X_train_seq_cov, class_feat=class_feat)
test_X = X_test_seq_cov.drop(class_feat, axis=1)
test_y = X_test_seq_cov[class_feat]
print("ripper score is ", clf.score(test_X, test_y))
