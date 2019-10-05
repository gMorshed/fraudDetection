import pandas as pd
import numpy as np
import io
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk

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
print(df.head())


fraud_labels = df["isFraud"]
df.drop(columns=["isFraud"], inplace=True)
print(df.head())
print(fraud_labels.head())

#Feature Selection
X_train, X_test, y_train, y_test = train_test_split(df, fraud_labels, test_size=0.2, random_state=0)
k=10
mi_transformer = SelectKBest(mutual_info_regression, k=4).fit(X_train, y_train)
mi_X_train = mi_transformer.transform(X_train)
mi_X_test = mi_transformer.transform(X_test)

