from django.shortcuts import render
from django.http import HttpResponse
import csv
from .test_skikit import Test_Skikit
from .model_operations import ModelOperations

# Create your views here.
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404
from django.forms import modelformset_factory

from zipfile import *
from .forms import ImageLabelForm, CreateMachineLearningModelForm, ImageBulkUploadForm, NumOfIterationForm
from .models import ImageLabel, MachineLearningModel, NumOfIteration
from django.shortcuts import get_object_or_404, render
import pandas as pd
import openpyxl
import os

from .ridgemodel import Model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import statistics
from matplotlib.backends.backend_agg import FigureCanvasAgg

from matplotlib import pylab
from pylab import *
import io, urllib, base64


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



class HomePageView(ListView):
    model = MachineLearningModel
    template_name = 'imagelabeling/home.html'


def IterationInputPage(request):

    model = NumOfIteration
    form_class = NumOfIterationForm
    template_name = 'imagelabeling/temp.html'
    return render(request, 'imagelabeling/temp.html',{'form': form_class})


def DisplayROC(request):
    path = '/Users/maggie/Desktop/active-learning/final_data_test.csv'
    data = Model.ridge_regression(path)
    #
    # newdf = Model.concateData(data)
    #
    # return HttpResponse(len(newdf))

    prediction = data['probability']
    actual = data['label'].values.reshape(-1, 1)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prediction)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC curve for Ridge Regression Model')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    return render(request, 'imagelabeling/graph.html', {'data': uri})

def CreatePostView(request):
    if request.method == 'POST':

        createMLModelForm = CreateMachineLearningModelForm(request.POST)

        if createMLModelForm.is_valid():
            create_model_form = createMLModelForm.save(commit=False)
            create_model_form.title = request.POST.get('title')
            create_model_form.save()

            # parse the zip file and create imageLabel objects
            print("about to handle files")
            # after that's done, we can train the model
            # redirect to model detail page
            return HttpResponseRedirect('/model/' + str(create_model_form.id) + '/upload')
        else:
            print(createMLModelForm.errors)
    else:
        createMLModelForm = CreateMachineLearningModelForm()
    return render(request, 'imagelabeling/post.html',
                  {'CreateMachineLearningModelForm': CreateMachineLearningModelForm})


def handle_uploaded_file(model, f):
    print("handling zip file")
    with ZipFile(f) as zip_file:
        # get the list of files
        names = zip_file.namelist()
        print("names")
        print(names)
        # handle your files as you need. You can read the file with:
        for name in names:
            image_file = zip_file.extract(name, "media/ml_model_images/" + model.title + "/")
            # hacky but need to reassign so correct path is assigned
            image_file = "ml_model_images/" + model.title + "/" + name
            photo = ImageLabel(machine_learning_model=model, image_file=image_file, title=name)
            photo.save()
            print("name again")


def bulk_upload_view(request, ml_model_id):
    try:
        ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    if request.method == 'POST':
        form = ImageBulkUploadForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(ml_model, request.FILES['bulk_upload'])
            return HttpResponseRedirect('/model/' + str(ml_model_id) + '/upload')
    else:
        form = ImageBulkUploadForm()
    return render(request, 'imagelabeling/bulk_upload_form.html',
                  {'BulkUploadForm': form, 'ml_model': ml_model})


def ml_model_detail(request, ml_model_id):
    try:
        ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
        images = ml_model.imagelabel_set.all()
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    return render(request, 'imagelabeling/model_detail.html', {'ml_model': ml_model, 'images': images})

def LabelImageView(request, ml_model_id):
    ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    # logic to send the image we want user to vote on
    desired_image_set = ml_model.imagelabel_set.all().order_by('adjusted_confidence')
    if len(desired_image_set) == 0:
        html = "<html><body>Currently no images</body></html>"
        return HttpResponse(html)
    desired_image = desired_image_set[0]
    return render(request, 'imagelabeling/label_image.html', {'desired_image': desired_image, 'ml_model': ml_model})


def image_label_detail(request, ml_model_id, image_id):
    try:
        ml_model = MachineLearningModel.get(pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    try:
        image = ImageLabel.objects.get(pk=image_id)
    except ImageLabel.DoesNotExist:
        raise Http404("Label does not exist")
    return render(request, 'imagelabeling/image_label_detail.html', {'image': image, 'ml_model': ml_model})


# this will just update our database with the user's vote of whether image is normal or abnormal
# we will envoke this from our form in /label, template label_image
def vote(request, image_id):
    image = get_object_or_404(ImageLabel, pk=image_id)
    ml_model = image.machine_learning_model
    ml_id = ml_model.id
    choice = request.POST['choice']
    if choice == "1":
        print("Choice is 1")
        selected_choice = image.one_votes
        selected_choice += 1
        image.one_votes = selected_choice
    elif choice == "0":
        print("Choice is 0")
        selected_choice = image.zero_votes
        selected_choice += 1
        image.zero_votes = selected_choice
    else:
        selected_choice = image.unknown_votes
        selected_choice += 1
        image.unknown_votes = selected_choice

    # recalculate confidence based on new vote
    if image.one_votes + image.zero_votes != 0:
        image.confidence = image.one_votes / (image.one_votes + image.zero_votes)
        # so we can sort by adjusted confidence level based on how sure abnormal vs normal it is
        image.adjusted_confidence = abs(image.confidence - 0.5)
    else:
        image.confidence = 0.5
        image.adjusted_confidence = 0
    image.save()
    return HttpResponseRedirect('/model/' + str(ml_id) + '/label')


def trainModel(request, ml_model_id):
    ml_model = get_object_or_404(ImageLabel, pk=ml_model_id)
    t = Test_Skikit()
    t.launch_training(ml_model.title)
    html = "<html><body>Training your model!</body></html>"
    return HttpResponse(html)


def testSkikit(request, ml_model_id):
    ml_model = get_object_or_404(ImageLabel, pk=ml_model_id)
    t = Test_Skikit()
    t.launch_predicting(ml_model.title)

    html = "<html><body>Testing the model against new data!</body></html>"
    return HttpResponse(html)


