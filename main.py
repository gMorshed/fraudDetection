import pandas as pd
testTran = pd.read_csv('transection_to_test_w_ground_truth.csv')
testTran.head(30)
testTran.drop(columns=['isFraud'],inplace=True)
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



number_of_fraud_trans = 500
number_of_non_fraud_trans = 500
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

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


#feature selection
k=5
mi_transformer = SelectKBest(mutual_info_regression,k=k).fit(X_train, y_train)
mi_X_train,mi_X_test = mi_transformer.transform(X_train), mi_transformer.transform(X_test)

testTran_selected_feature = testTran[testTran.columns[mi_transformer.get_support()]]
#code to make the test transection for the API
# import sys
# dataset = pd.DataFrame({'step': mi_X_test[:, 0], 'amount': mi_X_test[:, 1], 'oldbalanceOrg': mi_X_test[:, 2], 'newbalanceOrig': mi_X_test[:, 3], 'isMerchantDest': mi_X_test[:, 4]})
# dataset.drop(columns=['step'], inplace=True)
# dataset.head(10).to_csv('transection_to_test_w.csv', sep=',', encoding='utf-8',index=False)
# sys.exit()

for feature, importance in zip(fraud_features.columns, mi_transformer.scores_):
    print(f"The MI score for {feature} is {importance}")



'''
graphing for linear vs non linear data
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.scatter(X_train['amount'], X_train['oldbalanceOrg'], marker='o', c=y_train, s=25, edgecolor='k')
plt.title("2 classes of data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
'''


#feature Extraction
# pca_transformer = PCA(n_components=k).fit(X_train)
# pca_X_train = pca_transformer.transform(X_train)
# pca_X_test = pca_transformer.transform(X_test)

#need to add cross validation
#need to plot data to see if you need a kernel to separate the classes
#SVM doesnt need huge data set, so must reduce training examples to at least (10 * # features)
#Add: Regulariztion{Rudge Regression, Lasso, Elastic Net, Group Lasso} or sparistty induction to force the model to use less features

#SVM model
hyperparams = {
    "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4],
    "random_state": [0],
    'kernel':['linear','rbf','poly']
}
svc = SVC(gamma='auto')
clf = GridSearchCV(svc, hyperparams, scoring='accuracy',cv=5)
clf.fit(mi_X_train, y_train)
testScore = clf.score(mi_X_test, y_test)
print("SVM + Feature Selection Accuracy: ", testScore)
prediction_list = clf.predict(mi_X_test)
tn, fp, fn, tp = confusion_matrix(y_test,prediction_list).ravel()
print("tn, fp, fn, tp", tn, fp, fn, tp)
print("best params",clf.best_params_)
print(clf.predict(testTran_selected_feature.to_numpy()))
dump(clf, 'svm_selected_feature_with_kernal.joblib')

# optional = False
# if(optional):
#   C = 1.0  # SVM regularization parameter
#   models = (sk.svm.SVC(kernel='linear', C=C),
#     sk.svm.LinearSVC(C=C),
#     sk.svm.SVC(kernel='rbf', gamma=0.7, C=C),
#     sk.svm.SVC(kernel='poly', degree=3, C=C))
#   models = (clf.fit(X, y) for clf in models)


# svc2 = SVC(gamma='auto')
# clf2 = GridSearchCV(svc2, hyperparams, scoring='accuracy', cv=5)
# clf2.fit(pca_X_train, y_train)
# testScore2 = clf2.score(pca_X_test, y_test)
# print("SVM + Feature Extraction Accuracy: ", testScore2)


svc3 = SVC(gamma='auto')
clf3 = GridSearchCV(svc3, hyperparams, scoring='accuracy', cv=5)
clf3.fit(X_train, y_train)
testScore3 = clf3.score(X_test, y_test)
print("SVM with no Feature Selection/Extraction Accuracy: ", testScore3)
prediction_list = clf3.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,prediction_list).ravel()
print("tn, fp, fn, tp", tn, fp, fn, tp)
print("best params",clf3.best_params_)
print(clf3.predict(testTran.to_numpy()))

#dumping into pickel
dump(clf3, 'svm_all_feature_with_kernal.joblib')
