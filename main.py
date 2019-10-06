import pandas as pd
import numpy as np
# sompy taken from https://github.com/sevamoo/SOMPY
import sompy as sp
import io
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE


inputDataFrame = pd.read_csv('./data.csv')

#split test train
fraud_targets = pd.Series(inputDataFrame["isFraud"])
inputDataFrame.drop(columns=["isFraud"], inplace=True)
fraud_features = pd.DataFrame(inputDataFrame)
X_train, X_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)
#feature selection
# k=5
# mi_transformer = SelectKBest(mutual_info_regression,k=k).fit(X_train, y_train)
# mi_X_train,mi_X_test = mi_transformer.transform(X_train), mi_transformer.transform(X_test)
#
# for feature, importance in zip(fraud_features.columns, mi_transformer.scores_):
#     print(f"The MI score for {feature} is {importance}")

train_data = np.column_stack((X_train, y_train))

# Train SOM (Self-organizing-maps)
sm = sp.SOMFactory().build(train_data, normalization="var", initialization="random")
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

predicted = sm.predict(X_test, k=5, wt="distance")
print(predicted)

print(f"The accuracy score is {sklearn.metrics.accuracy_score(y_test, predicted)}")
print(f"The f1 score is {sklearn.metrics.f1_score(y_test, predicted)}")
