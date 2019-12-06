import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import sklearn as sk
import pickle
import sys
import io

if sys.argv[1] == "read":
	print("Reading data...")

	df = pd.read_csv('data.csv')
	df.head()
	df.dropna(inplace=True)

	assert df.isnull().sum().sum() == 0

	def f(name):
	  if(name[0] == "C"):
	    return 0
	  elif(name[0]=="M"):
	    return 1
	  else:
	    return NaN

	df["isMerchantOrig"] = df["nameOrig"].apply(func=f)
	df["isMerchantDest"] = df["nameDest"].apply(func=f)
	df.drop(columns=["nameOrig","nameDest"], inplace=True)
	new_df = pd.get_dummies(data=df["type"])

	for item in new_df:
	  df.insert(2, item,new_df[item]) 
	df.drop(columns=["type","isFlaggedFraud"], inplace=True)

	y = df["isFraud"]
	df.drop(columns=["isFraud"], inplace=True)
	X = df

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(sys.argv[2]), test_size=float(sys.argv[3]))

	print("Saving data...")
	pickle.dump(X_train, open('X_train.pkl', 'wb'))
	pickle.dump(X_test, open('X_test.pkl', 'wb'))
	pickle.dump(y_train, open('y_train.pkl', 'wb'))
	pickle.dump(y_test, open('y_test.pkl', 'wb'))
	pickle.dump(X, open('X.pkl', 'wb'))
	pickle.dump(y, open('y.pkl', 'wb'))
elif sys.argv[1] == "train":
	print("Loading data...")
	X_train = pickle.load(open('X_train.p', 'rb'))
	X_test = pickle.load(open('X_test.p', 'rb'))
	y_train = pickle.load(open('y_train.p', 'rb'))
	y_test = pickle.load(open('y_test.p', 'rb'))

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,), random_state=1)

	clf.fit(X_train, y_train)               
	# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
	pickle.dump(clf, open("model.p", 'wb'))
	# pickle.dump(y_train, open('y_train.p', 'wb'))
	# pickle.dump(y_test, open('y_test.p', 'wb'))
	# pickle.dump(X_train, open('X_train.p', 'wb'))
	# pickle.dump(X_test, open('X_test.p', 'wb'))
elif sys.argv[1] == "test":
	print("Loading data...")
	# X_test = pickle.load(open('X_test.pkl', 'rb'))
	# y_test = pickle.load(open('y_test.pkl', 'rb'))
	# X = pickle.load(open('X.pkl', 'rb'))
	# y = pickle.load(open('y.pkl', 'rb'))
	X_test = pickle.load(open('X_test.p', 'rb'))
	y_test = pickle.load(open('y_test.p', 'rb'))
	X_train = pickle.load(open('X_train.p', 'rb'))
	y_train = pickle.load(open('y_train.p', 'rb'))
	clf = pickle.load(open('model.p', 'rb'))
	# print(clf.score(X_test, y_test))
	# print(X_test)
	# print(y_test)
	# print(metrics.f1_score(y_test, X_test))
	print(metrics.classification_report(clf.predict(X_test), y_test))
	print(metrics.confusion_matrix(clf.predict(X_test), y_test))
	# print(metrics.accuracy_score(y_test, X_test.round(), normalize=False))
	# print(metrics.confusion_matrix(y_test, X_test))
	from sklearn.model_selection import cross_val_score
	# print("Cross validating...")
	# print(cross_val_score(clf, X, y, cv=5))
elif sys.argv[1] == "kfold":
	print("Loading data...")
	X = pickle.load(open('X.pkl', 'rb'))
	y = pickle.load(open('y.pkl', 'rb'))
	clf = pickle.load(open('model.p', 'rb'))
	# print(clf.score(X_test, y_test))
	from sklearn.model_selection import cross_val_score
	print("Testing with", sys.argv[2], 'folds...')
	print(cross_val_score(clf, X, y, cv=int(sys.argv[2]), n_jobs=1))
else:
	print("Error:", sys.argv[1])