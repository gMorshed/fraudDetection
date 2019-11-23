# Create your views here.
import csv
import io

from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
import pandas as pd


# Create your views here.
# one parameter named request
def profile_upload(request):
    expected_features = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest',
                         'oldbalanceDest',
                         'newbalanceDest', 'isFraud', 'isFlaggedFraud']
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

    df = pd.DataFrame(rows, columns=features, dtype=float)
    print(df.head())

    context = {}
    return HttpResponse("Dataframe looks good. Let's do machine learning.")
