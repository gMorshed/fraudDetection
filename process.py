import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

raw_data = loadarff('credit_fraud.arff')
df = pd.DataFrame(raw_data[0])

train, test = train_test_split(df, test_size=0.1)

print(train.head())
print(test.head())
