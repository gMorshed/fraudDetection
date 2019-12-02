import pandas as pd
import numpy as np
# sompy taken from https://github.com/sevamoo/SOMPY
import sompy as sp
# minisom taken from https://github.com/JustGlowing/minisom
import minisom as ms
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import io
import pickle
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE

# ******************** Train on synthetic dataset ******************************
inputDataFrame = pd.read_csv('./data.csv')

#split test train
fraud_targets = pd.Series(inputDataFrame["isFraud"])
inputDataFrame.drop(columns=["isFraud"], inplace=True)
fraud_features = pd.DataFrame(inputDataFrame)
X_train, X_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

# just a heuristic for x and y as it needs to be constant to tune the other hyperparameters
x = int(np.sqrt(5 * np.sqrt(fraud_features.shape[0])))
y = x
# number of features
input_len = fraud_features.shape[1]

space = {'sig': hp.uniform('sig', 0.001, x / 2.01), 'learning_rate': hp.uniform('learning_rate', 0.001, 5)}

# taken from example on https://github.com/JustGlowing/minisom/blob/master/examples/Classification.ipynb
def classify(som, data, class_assignments):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = class_assignments
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def som_quantization_error(space):
    sig = space['sig']
    learning_rate = space['learning_rate']
    error = ms.MiniSom(x=x, y=y, input_len=input_len, sigma=sig, learning_rate=learning_rate).quantization_error(X_train)
    return {'loss': error, 'status': STATUS_OK}

# hyperparameter tuning to obtain sigma and learning rate
trials = Trials()
best = fmin(fn=som_quantization_error, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print(best)

som = ms.MiniSom(x=x, y=y, input_len=input_len, sigma=8.007684739287342, learning_rate=4.486348532872689)
som.pca_weights_init(X_train)
som.train_batch(X_train, 100)
class_assignments = som.labels_map(X_train, y_train)

print(sklearn.metrics.classification_report(y_test, classify(som, X_test, class_assignments)))

# saving the som in the file som.p
# with open('synthetic_som.p', 'wb') as outfile:
#     pickle.dump(som, outfile)

#feature selection
# k=5
# mi_transformer = SelectKBest(mutual_info_regression,k=k).fit(X_train, y_train)
# mi_X_train,mi_X_test = mi_transformer.transform(X_train), mi_transformer.transform(X_test)
#
# for feature, importance in zip(fraud_features.columns, mi_transformer.scores_):
#     print(f"The MI score for {feature} is {importance}")

# train_data = np.column_stack((X_train, y_train))
#
# # Train SOM (Self-organizing-maps)
# sm = sp.SOMFactory().build(train_data, normalization="var", initialization="random")
# sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)
#
# predicted = sm.predict(X_test, k=5, wt="distance")
# print(predicted)
#
# print(f"The accuracy score is {sklearn.metrics.accuracy_score(y_test, predicted)}")
# print(f"The f1 score is {sklearn.metrics.f1_score(y_test, predicted)}")
