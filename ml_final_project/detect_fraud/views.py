# Create your views here.
import csv
import io

from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix


def classify_fradulent(predict):
    if predict:
        return "Fraudulent Transaction"
    return "Non Fraudulent Transaction"


def run_fraud_predict(dataframe, model):
    clf = None
    if model == "Random forest":
        clf = load('../random_forest_w_all_feature.joblib')
    elif model == "SVM":
        clf = load('../svm_all_feature_with_kernal.joblib')
    elif model == "Self Organizing Map":
        clf = load('../random_forest_w_all_feature.joblib')
    else:
        clf = load('../random_forest_w_all_feature.joblib')
    dataframe.rename(columns={"isFraud": "isFraud_ground_truth"}, inplace=True)
    holding_ground_truth = dataframe['isFraud_ground_truth']
    dataframe.drop(columns=["isFraud_ground_truth"], inplace=True)
    prediction_list = clf.predict(dataframe.to_numpy())
    dataframe = dataframe.join(holding_ground_truth)
    dataframe.insert(14, "fraud_prediction", prediction_list, True)
    tn, fp, fn, tp = confusion_matrix(dataframe['isFraud_ground_truth'].to_numpy(), prediction_list).ravel()
    confusion_matrix_simplify = ["True Negative: " + str(tn), "False Positive: " + str(fp),
                                 "False Negative: " + str(fn), "True Positive: " + str(tp)]

    report = classification_report(dataframe['isFraud_ground_truth'].to_numpy(), prediction_list,
                                   target_names=['Non-fraud', 'Fraud'],output_dict=True)
    dataframe["fraud_prediction"] = dataframe["fraud_prediction"].apply(func=classify_fradulent)
    dataframe["isFraud_ground_truth"] = dataframe["isFraud_ground_truth"].apply(func=classify_fradulent)
    return dataframe, confusion_matrix_simplify, report


# Create your views here.
# one parameter named request
def profile_upload(request):
    expected_features = ['step', 'TRANSFER', 'PAYMENT', 'DEBIT', 'CASH_OUT', 'CASH_IN', 'amount', 'oldbalanceOrg',
                         'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isMerchantOrig', 'isMerchantDest',
                         'isFraud']
    # declaring template
    template = "profile_upload.html"
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template)
    csv_file = request.FILES.get(u'csv_file')
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'File is not CSV type')
        return render(request, template)

    # if file is too large, return
    if csv_file.multiple_chunks():
        messages.error(request, "Uploaded file is too big (%.2f MB)." % (csv_file.size / (1000 * 1000),))
        return render(request, template)

    # now reading the data
    data_set = csv_file.read().decode('UTF-8')
    features = []
    rows = []
    first_col = True
    for data in data_set.split('\n'):
        if first_col:
            for item in data.split(','):
                features.append(item.strip('\r'))
            first_col = False
        else:
            r = []
            for item in data.split(','):
                r.append(item)
            rows.append(r)

    # feature validation
    for feature in features:
        if not (feature in expected_features):
            print(feature in expected_features)
            messages.error(request, 'File should not contain this column: %s' % feature)
            messages.error(request, 'It should contain these columns: %s' % expected_features)
            return render(request, template)

    model  = request.POST['model']
    df = pd.DataFrame(rows, columns=features, dtype=float)
    df = df.dropna()
    # render dataframe as html
    result_df, confusion_matrix_simplify, report = run_fraud_predict(df, model)
    html = result_df.to_html()
    # write html to file
    text_file = open("templates/table.html", "w")
    text_file.write(html)
    text_file.close()

    report_dataframe = pd.DataFrame(report).transpose()
    html = report_dataframe.to_html()
    # write html to file
    text_file = open("templates/report.html", "w")
    text_file.write(html)
    text_file.close()

    return render(request, 'result.html', {'matrix_result': confusion_matrix_simplify, 'model_name': model})
