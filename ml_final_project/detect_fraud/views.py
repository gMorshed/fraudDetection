# Create your views here.
import csv
import io

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd

from ml_final_project import settings
from .models import *


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


# Create your views here.
# one parameter named request
def profile_upload(request):
    # declaring template
    template = "profile_upload.html"
    data = Profile.objects.all()
    # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address,    phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES.get(u'file')
    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')

    features = []
    rows = []
    first_col = True
    for data in data_set.split('\n'):
        if first_col:
            for item in data.split(','):
                features.append(item)
            first_col = False
        else:
            r = []
            for item in data.split(','):
                r.append(item)
            rows.append(r)

    df = pd.DataFrame(rows, columns=features, dtype=float)
    print(df.head())

    context = {}
    return HttpResponse("Hello, world. You're at the polls index.")
