import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

raw_data = loadarff('credit_fraud.arff')
df = pd.DataFrame(raw_data[0])

print(df.head())

df.dropna(inplace=True)
print(df.head())

# Create dummy columns for all qualitative columns and replace old columns
for column in ["over_draft", "credit_history", "purpose", "Average_Credit_Balance",
               "employment", "personal_status", "other_parties", "property_magnitude",
               "housing", "job", "foreign_worker"]:
    new_inputDataFrame = pd.get_dummies(data=df[column])
    for item in new_inputDataFrame:
        df.insert(2, item,new_inputDataFrame[item])
    df.drop(columns=[column], inplace=True)

print(df.head())

# Convert other_payment_plans feature to dummies manually because a field name
# duplicates a field name used in a different feature
def getPaymentPlanDummies(oldVal):
    if oldVal == b'bank':
        return 0
    elif oldVal == b'stores':
        return 1
    else:
        return 2

df["other_payment_plans"] = df["other_payment_plans"].apply(func=getPaymentPlanDummies)

def getOwnTelephoneDummies(oldVal):
    if oldVal == b'none':
        return 0
    else:
        return 1

df["own_telephone"] = df["own_telephone"].apply(func=getOwnTelephoneDummies)

# Convert labels to integer (0 for non-fraudulent, 1 for fraudulent)
def getNewLabel(oldLabel):
    if oldLabel == b'good':
        return 0
    else:
        return 1

df["class"] = df["class"].apply(func=getNewLabel)
print(df.head())

df.to_csv("./german_data.csv")

# Split test train
fraud_targets = pd.Series(df["class"])
df.drop(columns=["class"], inplace=True)
fraud_features = pd.DataFrame(df)
X_train, X_test, y_train, y_test = train_test_split(fraud_features, fraud_targets, test_size=0.2, random_state=0)
