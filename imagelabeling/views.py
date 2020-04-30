from django.shortcuts import render
from django.http import HttpResponse
import csv
from .test_skikit import Test_Skikit
from .model_operations import ModelOperations
from .model_operations_numbers import ModelOperationsNumbers

# get all models
from django.db import models

# views here.
from django.views.generic import ListView, CreateView
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404

# for turning queryset to json
from django.core import serializers

# zipfile for handling bulk upload
from zipfile import *

# get the forms
from .forms import ImageLabelForm, CreateMachineLearningModelForm, CreateMachineLearningNumbersModelForm, ImageBulkUploadForm, GammaForm, BooleanForm, SVMKernel, CreateDynamicModelForm, DigitFeatureForm, NumShapeForm, AlphaInputForm
from .models import ImageLabel, MachineLearningModel, MachineLearningNumbersModel, ModelSchema, FieldSchema, NumberLabel,  DigitFeature, AlphaInput
from django.shortcuts import get_object_or_404, render

# ml stuff
import pandas as pd

from .ridgemodel import Calculation
# from .ridgemodel_multiclass import Calculation as CalculationMultiClass

from .ridgemodel_multiclass import CalculationMultiClass
from pylab import *
import io, urllib, base64
from django.db.models import F
from sklearn.metrics import roc_curve, auc

# get models
from django.apps import apps
import re


class HomePageView(ListView):
    model = MachineLearningModel
    template_name = 'imagelabeling/home.html'

## form to ask user to input alhpa value for ridge regression
def GetAlpha(request,ml_model_id):

    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    ml_id = ml_model.id
    if request.method == "POST":


        form = AlphaInputForm(request.POST)
        form.save()

    else:
        form = AlphaInputForm

    return render(request, "imagelabeling/alpha_input.html", {'form' : form,'ml_model':ml_model})


def userTuneRidge(request,ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    if request.method == "GET":

        ## get all alpha value in database
        allA = AlphaInput.objects.all()

        ## get the latest alpha to train model
        a_obj = AlphaInput.objects.last()

        ## get the alpha value
        a_value = a_obj.alpha_input

        # print(avalue.alpha_input)

        ## read in the latest csv
        path = "/Users/maggie/Desktop/active-learning/final_data_test_relabel_demo.csv"
        df = Calculation.readCSV(path)

        ## run ridge regression
        result = Calculation.RidgeWithTuning(df,a_value)
        print(result[1])

        ## r^2 score
        r_score = result[2]

        ## generate ROC curve

        plt = Calculation.ROC(result[0])
        fig = plt.gcf()
        buf1 = io.BytesIO()
        fig.savefig(buf1, format='png')
        buf1.seek(0)
        string1 = base64.b64encode(buf1.read())
        uri1 = urllib.parse.quote(string1)

        return render(request, 'imagelabeling/a_Value.html', {"uri1": uri1, 'a_obj': a_obj, 'r_score': r_score, 'ml_model':ml_model})
    else:
        return HttpResponse("not work")


# create a new vote to redirect to uncertain cases
def Relabelvote(request, image_id):
    df = pd.read_csv("/Users/maggie/Desktop/active-learning/final_data_test_relabel_demo.csv")

    # image = get_object_or_404(ImageLabel, pk=image_id)
    image = ImageLabel.objects.get(pk = image_id)
    ml_model = image.machine_learning_model
    ml_id = ml_model.id
    choice = request.POST['choice']
    if choice == "1":
        print("Choice is 1")
        selected_choice = image.one_votes
        selected_choice += 1
        image.one_votes = selected_choice
        image.model_classification = 1
    elif choice == "0":
        print("Choice is 0")
        selected_choice = image.zero_votes
        selected_choice += 1
        image.zero_votes = selected_choice
        image.model_classification = 0
    else:
        selected_choice = image.unknown_votes
        selected_choice += 1
        image.unknown_votes = selected_choice

    # recalculate confidence based on new vote
    if image.one_votes + image.zero_votes != 0:
        image.user_score = image.one_votes / (image.one_votes + image.zero_votes)
        # so we can sort by adjusted confidence level based on how sure abnormal vs normal it is
        image.adjusted_user_score = abs(image.user_score - 0.5)
    else:
        image.user_score = 0.5
        image.adjusted_user_score = 0
    image.save()
    # change the original label with new label entered by user

    #get the new label
    newlabel = image.model_classification
    #get the title of iamge
    t = image.title

    #find the image with approriate title
    df1 = df[df['image'].str.contains(t)]

    #find the row index of the image
    r_index = df1.index.values.astype(int)[0]

    #update the label of the image
    df.iloc[[r_index], [1001]] = newlabel

    # print(df.iloc[[r_index], [1001]])

    #overwrite the csv file
    df.to_csv('final_data_test_relabel_demo.csv', index=False)

    return HttpResponseRedirect('/model/' + str(ml_id) + '/f/')
    # return HttpResponseRedirect(request.META.get('HTTP_REFERER'))



def jump(request,ml_model_id, image_id):

    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    image = ImageLabel.objects.get(pk=image_id)
    label_form = ImageLabelForm(request.POST)

    return render(request, 'imagelabeling/jump_detail.html',{'image':image,'label_form':label_form,'ml_model':ml_model})


def filter(request, ml_model_id):

    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    # print(ml_model_id)

    clean_uncertain_list = []
    full_name_uncertain = []
    id_tuple =[]

    #read csv file generated by firss
    df = pd.read_csv("/Users/maggie/Desktop/active-learning/final_data_test_relabel_demo.csv")
    low_dif = df[df['dif'] < 0.3]
    uncertain_list = low_dif['image']
    # print(uncertain_list)


    for elem in uncertain_list:
        tmp = elem.split("/")
        clean_uncertain_list.append(tmp[4])
        tmp_full = tmp[3] + "/" + tmp[4]
        full_name_uncertain.append(tmp_full)


    for elem in clean_uncertain_list:
            elem_id = ImageLabel.objects.only('id').get(title__contains=elem).id
            id_tuple.append(elem_id)

    # print(id_tuple)

    # extract the objects with near 0.5 probability using ID
    obj = ImageLabel.objects.filter(id__in=id_tuple)
    # print(obj)

    args = {'obj': obj,'ml_model':ml_model}
    # # change model_classification by id
    # id = 343
    # ImageLabel.objects.filter(id=id).update(model_classification=0)
    # s = ImageLabel.objects.get(id = id)

    # # extract the objects with near 0.5 probability using title
    # obj = ImageLabel.objects.filter(title__in=full_name_uncertain)
    # # print(obj)
    #
    # # # change model_classification by title
    # t = full_name_uncertain[0]
    # ImageLabel.objects.filter(title=t).update(model_classification=0)
    #
    # s = ImageLabel.objects.get(title=t)
    # newlabel = s.model_classification
    #
    # #find the row index that is uncertain
    # df1 = df[df['image'].str.contains(t)]
    # r_index = df1.index.values.astype(int)[0]
    #
    # df.iloc[[r_index],[1001]] = newlabel
    #
    # print(df.iloc[[r_index],[1001]])
    #
    # df.to_csv('final_data_test_relabel_demo_modified.csv', index=False)

    return render(request, 'imagelabeling/f.html', args)

    # return HttpResponse("work")

def tryLabel(request):
    images = get_object_or_404(NumberLabel)
    print(len(images))
    return HttpResponse("count of image")

# the function is used to test the multiclass classifier
def CalculateProbabilityNum(request):
    path = "/Users/maggie/Desktop/active-learning/digit_features.csv"
    df = CalculationMultiClass.readCSV(path)
    multi_model_df = CalculationMultiClass.oneVSrest(df)

    print(multi_model_df['probability'])
    print(multi_model_df['predicted_label'])

    return HttpResponse("work")

# get save form in database, convert into csv for testing
def getFeaturefromDB():
    myModel = DigitFeature.objects.all()
    # myModel = get_object_or_404(DigitFeature)

    total_digit_col = []
    horizontal_col = []
    vertical_col = []
    loops_col = []
    close_eye_col = []
    open_eye_col = []
    acute_col = []
    right_col = []
    # curve_col = []
    label_col = []
    df = pd.DataFrame()
    for digit in myModel:
        total_digit_col.append(digit.total_digit)
        horizontal_col.append(digit.horizontal_line)
        vertical_col.append(digit.vertical_line)
        loops_col.append(digit.loops)
        close_eye_col.append(digit.close_eye_hook)
        open_eye_col.append(digit.open_eye_hook)
        acute_col.append(digit.acute_Angle)
        right_col.append(digit.right_Angle)
        # curve_col.append(digit.curve)
        label_col.append(digit.label)


    df['total_digit'] = total_digit_col
    df['horizontal_line'] = horizontal_col
    df['vertical_line'] = vertical_col
    df['loops'] = loops_col
    df['close_eye_hook'] = close_eye_col
    df['open_eye_hook'] = open_eye_col
    df['acute_angle'] = acute_col
    df['right_angle'] = right_col
    # df['curve'] = curve_col
    df['label'] = label_col

    df = df.to_csv('feature_db2.csv',index=False)

    return df





## form to ask user to input features of numbers
def register(request,ml_model_id,image_id):
    try:
        ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    try:
        image = ImageLabel.objects.get(pk=image_id)
    except ImageLabel.DoesNotExist:
        raise Http404("Label does not exist")
    return render(request, 'imagelabeling/register.html', {'image': image, 'ml_model': ml_model})

    # and request.POST
    # feature_form = DigitFeatureForm(data=request.POST)
    # image = NumberLabel.objects.get()
    # if request.method == "POST":
    #     # print(image)
    #
    #     if feature_form.is_valid():
    #         total_digit = feature_form.cleaned_data["total_digit"]
    #         horizontal_line = feature_form.cleaned_data["horizontal_line"]
    #         vertical_line = feature_form.cleaned_data["vertical_line"]
    #         loops = feature_form.cleaned_data["loops"]
    #         close_eye_hook = feature_form.cleaned_data["close_eye_hook"]
    #         open_eye_hook = feature_form.cleaned_data["open_eye_hook"]
    #         acute_Angle = feature_form.cleaned_data["acute_Angle"]
    #         right_Angle = feature_form.cleaned_data["right_Angle"]
    #         # curve = feature_form.cleaned_data["curve"]
    #         label = feature_form.cleaned_data['label']
    #
    #         DigitFeature.objects.create(total_digit=total_digit,horizontal_line=horizontal_line,loops=loops,
    #                                     vertical_line=vertical_line,close_eye_hook=close_eye_hook,open_eye_hook=open_eye_hook,
    #                                     acute_Angle=acute_Angle,right_Angle=right_Angle,label=label)
    #         feature_form = DigitFeatureForm()
    #
    # else:
    #     feature_form = DigitFeatureForm()
    # return render(request, "imagelabeling/register.html",{'feature_form': feature_form})


#
def labelDigitFeatures(request, ml_model_id, numbers_image_id):
    # get the picture of the image to display above the form
    try:
        ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    except MachineLearningNumbersModel.DoesNotExist:
        raise Http404("Model does not exist")
    try:
        image = NumberLabel.objects.get(pk=numbers_image_id)
    except NumberLabel.DoesNotExist:
        raise Http404("Label does not exist")

    if request.method == "POST":
        feature_form = DigitFeatureForm(request.POST)
        shape_form = NumShapeForm(request.POST)
    else:
        feature_form = DigitFeatureForm
        shape_form = NumShapeForm

    # return HttpResponse("hello")
    return render(request, 'imagelabeling/label_digit.html',{'feature_form':feature_form, 'shape_form':shape_form, 'ml_model': ml_model, 'image': image})

def SVMTuning(request, ml_model_id):
    if request.method == "POST":
        form = GammaForm(request.POST)
        form2 = SVMKernel(request.POST)

        if form.is_valid() and form2.is_valid():
            return HttpResponse("gamma & kernel works")
    else:
        form = GammaForm
        form2 = SVMKernel

    return render(request, 'imagelabeling/svm_selection.html', {'form': form, 'form2':form2})


# call this view right after this model is created
def CalculateProbability2(request, ml_model_id):


    # get name of model we're running analysis on
    try:
        ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")

    path = 'final_data_test_' + ml_model.title + '.csv'
    # path = '/Users/maggie/Desktop/active-learning/large_data.csv'
    firstdf = Calculation.readCSV(path)

    count = 0

    if request.method == "GET":
        bestl1 = Calculation.best_lambda(firstdf)
        rid_result = Calculation.ridge_regression(firstdf,bestl1)
        concatedDF = Calculation.concateData(rid_result)
        bestl2 = Calculation.best_lambda(concatedDF)
        final = Calculation.ridge_regression(concatedDF,bestl2)

        print("       ")
        print(len(final['probability']))
        print("          ")

        Calculation.outputCSV(final, ml_model.title)
        Calculation.outputJSON(final, ml_model.title)

        plt = Calculation.ROC(final)
        fig = plt.gcf()

        buf1 = io.BytesIO()
        fig.savefig(buf1, format='png')
        buf1.seek(0)
        string1 = base64.b64encode(buf1.read())
        uri1 = urllib.parse.quote(string1)

        form_class = BooleanForm
        args = {'image': uri1, 'form': form_class}

        # return HttpResponseRedirect('/model/' + str(ml_model_id) + '/run-predictions')
        # return HttpResponseRedirect('/model/' + str(ml_model_id) + '/probability')
        return render(request, 'imagelabeling/graph.html', {'image': uri1})

def CalculateProbability(request):

    # get name of model we're running analysis on
    # try:
    #     ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    # except MachineLearningModel.DoesNotExist:
    #     raise Http404("Model does not exist")

    path = '/Users/maggie/Desktop/active-learning/final_data_test_xray.csv'
    firstdf = Calculation.readCSV(path)
    count = 0

    if request.method == "GET":

    # path = 'final_data_test_' + ml_model.title + '.csv'
    # else:

        # final = Calculation.newridge(firstdf)

        print("original", len(firstdf))

        # if request.method == "GET":
        # form_class = BooleanForm
        # # print(form_class)

        best_alpha = Calculation.best_lambda(firstdf)

        rid_result = Calculation.ridge_regression(firstdf,best_alpha)

        print("low diff", len(Calculation.low_diff(rid_result)))

        concatedDF = Calculation.concateData(rid_result)
        # final = Calculation.ridge_regression(concatedDF,best_alpha)
        #
        print("       ")
        print("after concate", len(concatedDF['probability']))
        print("          ")

        #
        Calculation.outputCSV(concatedDF)
        # Calculation.outputCSV(final, ml_model.title)
        # Calculation.outputJSON(final, ml_model.title)

        # plt = final
        plt = Calculation.ROC(concatedDF)
        fig = plt.gcf()

        buf1 = io.BytesIO()
        fig.savefig(buf1, format='png')
        buf1.seek(0)
        string1 = base64.b64encode(buf1.read())
        uri1 = urllib.parse.quote(string1)

        return render(request, 'imagelabeling/graph.html', {'image': uri1})






            # return HttpResponseRedirect('/model/' + str(ml_model_id) + '/run-predictions')




def CalculateProbabilityNumbers(request, ml_model_id):
    # get name of model we're running analysis on
    try:
        ml_model = MachineLearningNumbersModel.objects.get(pk=ml_model_id)
    except MachineLearningNumbersModel.DoesNotExist:
        raise Http404("Model does not exist")

    path = 'final_data_' + ml_model.title + '.csv'

    #path = '/Users/maggie/Desktop/active-learning/large_data.csv'
    firstdf = CalculationMultiClass.readCSV(path)



    count = 0

    if request.method == "GET":

            # need to figure out what this is supposed to do in regards to ridge regression model
            rid_result = CalculationMultiClass.oneVsOne(firstdf)
            concatedDF = CalculationMultiClass.concateData(rid_result)
            final = CalculationMultiClass.oneVsOne(concatedDF)

            print("       ")
            print(len(final['probability']))
            print("          ")

            CalculationMultiClass.outputCSVCM(final, ml_model.title)
            CalculationMultiClass.outputJSON(final, ml_model.title)
            return HttpResponseRedirect('/model/' + str(ml_model_id) + '/run-predictions-numbers')

def RenderGraph(request):
    return render(request, 'imagelabeling/graph.html')




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


def CreateNumbersModelView(request):
    if request.method == 'POST':

        createMLModelForm = CreateMachineLearningNumbersModelForm(request.POST)

        if createMLModelForm.is_valid():
            create_model_form = createMLModelForm.save(commit=False)
            create_model_form.title = request.POST.get('title')
            create_model_form.save()

            # parse the zip file and create imageLabel objects
            print("about to handle files")
            # after that's done, we can train the model
            # redirect to model detail page
            return HttpResponseRedirect('/model/' + str(create_model_form.id) + '/upload-numbers')
        else:
            print(createMLModelForm.errors)
    else:
        createMLModelForm = CreateMachineLearningNumbersModelForm()
    return render(request, 'imagelabeling/post.html',
                  {'CreateMachineLearningModelForm': CreateMachineLearningNumbersModelForm})


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
            return HttpResponseRedirect('/model/' + str(ml_model_id) + '/train')
    else:
        form = ImageBulkUploadForm()
    return render(request, 'imagelabeling/bulk_upload_form.html',
                  {'BulkUploadForm': form, 'ml_model': ml_model})

def bulk_upload_view_number(request, ml_model_id):
    try:
        ml_model = MachineLearningNumbersModel.objects.get(pk=ml_model_id)
    except MachineLearningNumbersModel.DoesNotExist:
        raise Http404("Model does not exist")
    if request.method == 'POST':
        form = ImageBulkUploadForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file_number(ml_model, request.FILES['bulk_upload'])
            return HttpResponseRedirect('/model/' + str(ml_model_id) + '/train-numbers')
    else:
        form = ImageBulkUploadForm()
    return render(request, 'imagelabeling/bulk_upload_form.html',
                  {'BulkUploadForm': form, 'ml_model': ml_model})

def handle_uploaded_file_number(model, f):
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
            photo = NumberLabel(machine_learning_model=model, image_file=image_file, title=name)
            photo.save()
            print("name again")

# this trains the model once, then should redirect to iteration page i think
def trainModel(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    t = ModelOperations()
    # change from the multithreaded solution since that might be breaking
    t.train_model(ml_model.title)
    return HttpResponseRedirect('/model/' + str(ml_model_id) + '/probability')


def ml_model_detail(request, ml_model_id):
    try:
        ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
        images = ml_model.imagelabel_set.all()
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    return render(request, 'imagelabeling/model_detail.html', {'ml_model': ml_model, 'images': images})

#
# def LabelImageView(request, ml_model_id):
#     ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
#     # logic to send the image we want user to vote on
#     desired_image_set = ml_model.imagelabel_set.all().order_by('adjusted_user_score')
#     if len(desired_image_set) == 0:
#         html = "<html><body>Currently no images</body></html>"
#         return HttpResponse(html)
#     desired_image = desired_image_set[0]
#     return render(request, 'imagelabeling/label_image.html', {'desired_image': desired_image, 'ml_model': ml_model})


def image_label_detail(request, ml_model_id, image_id):
    try:
        ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    try:
        image = ImageLabel.objects.get(pk=image_id)
    except ImageLabel.DoesNotExist:
        raise Http404("Label does not exist")
    return render(request, 'imagelabeling/image_label_detail.html', {'image': image, 'ml_model': ml_model})


# this will just update our database with the user's vote of whether image is normal or abnormal
# we will envoke this from our form in /label, template label_image
# def vote(request, image_id):
#     image = get_object_or_404(ImageLabel, pk=image_id)
#     ml_model = image.machine_learning_model
#     ml_id = ml_model.id
#     choice = request.POST['choice']
#     if choice == "1":
#         print("Choice is 1")
#         selected_choice = image.one_votes
#         selected_choice += 1
#         image.one_votes = selected_choice
#         image.model_classification = 1
#     elif choice == "0":
#         print("Choice is 0")
#         selected_choice = image.zero_votes
#         selected_choice += 1
#         image.zero_votes = selected_choice
#         image.model_classification = 0
#     else:
#         selected_choice = image.unknown_votes
#         selected_choice += 1
#         image.unknown_votes = selected_choice
#
#     # recalculate confidence based on new vote
#     if image.one_votes + image.zero_votes != 0:
#         image.user_score = image.one_votes / (image.one_votes + image.zero_votes)
#         # so we can sort by adjusted confidence level based on how sure abnormal vs normal it is
#         image.adjusted_user_score = abs(image.user_score - 0.5)
#     else:
#         image.user_score = 0.5
#         image.adjusted_user_score = 0
#     image.save()
#     return HttpResponseRedirect('/model/' + str(ml_id) + '/label')

# update image_label object with classification and probability
def updateImagesWithModelPrediction(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    images = ml_model.imagelabel_set.all()
    df = pd.read_csv('final_data_test_' + ml_model.title + '.csv')
    for image in images:
        title = "media/ml_model_images/" + ml_model.title + "/" + image.title
        entry = df.loc[df['image'] == title]

        # check for broken image labels or duplicates, needs to be 1
        if len(entry) == 1:
            image.model_score = entry["label"]
            image.model_difference = entry["dif"]
            image.model_probability = entry["probability"]
            image.save()
    return HttpResponseRedirect('/model/' + str(ml_model_id) + '/visualization')


def visualization(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    images = ml_model.imagelabel_set.all()
    images_json = serializers.serialize('json', images)
    # so from each image, we need the adjusted prediction number by the model
    # then the number given by the labelers
    return render(request, 'imagelabeling/visualizations.html', {'images': images_json, 'ml_model': ml_model})


def ml_numbers_model_detail(request, ml_model_id):
    try:
        ml_model = MachineLearningNumbersModel.objects.get(pk=ml_model_id)
        images = ml_model.imagelabel_set.all()
    except MachineLearningNumbersModel.DoesNotExist:
        raise Http404("Model does not exist")
    return render(request, 'imagelabeling/model_detail.html', {'ml_model': ml_model, 'images': images})


def numbers_image_label_detail(request, ml_model_id, numbers_image_id):
    try:
        ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    except MachineLearningNumbersModel.DoesNotExist:
        raise Http404("Model does not exist")
    try:
        image = NumberLabel.objects.get(pk=numbers_image_id)
    except NumberLabel.DoesNotExist:
        raise Http404("Label does not exist")
    return render(request, 'imagelabeling/number_image_label_detail.html', {'image': image, 'ml_model': ml_model})

#
# this will just update our database with the user's vote of whether image is normal or abnormal
# we will envoke this from our form in /label, template label_image
def voteNumbers(request, image_id):
    image = get_object_or_404(NumberLabel, pk=image_id)
    ml_model = image.machine_learning_model
    ml_id = ml_model.id
    choice = request.POST['choice']
    if choice == "0":
        print("Choice is 0")
        selected_choice = image.zero_votes
        selected_choice += 1
        image.zero_votes = selected_choice
    elif choice == "1":
        print("Choice is 1")
        selected_choice = image.one_votes
        selected_choice += 1
        image.one_votes = selected_choice
    elif choice == "2":
        print("Choice is 2")
        selected_choice = image.two_votes
        selected_choice += 1
        image.two_votes = selected_choice
    elif choice == "3":
        print("Choice is 3")
        selected_choice = image.three_votes
        selected_choice += 1
        image.three_votes = selected_choice
    elif choice == "4":
        print("Choice is 4")
        selected_choice = image.four_votes
        selected_choice += 1
        image.four_votes = selected_choice
    elif choice == "5":
        print("Choice is 5")
        selected_choice = image.five_votes
        selected_choice += 1
        image.five_votes = selected_choice
    elif choice == "6":
        print("Choice is 6")
        selected_choice = image.six_votes
        selected_choice += 1
        image.six_votes = selected_choice
    elif choice == "7":
        print("Choice is 7")
        selected_choice = image.seven_votes
        selected_choice += 1
        image.seven_votes = selected_choice
    elif choice == "8":
        print("Choice is 8")
        selected_choice = image.eight_votes
        selected_choice += 1
        image.eight_votes = selected_choice
    elif choice == "9":
        print("Choice is 9")
        selected_choice = image.nine_votes
        selected_choice += 1
        image.nine_votes = selected_choice
    else:
        selected_choice = image.unknown_votes
        selected_choice += 1
        image.unknown_votes = selected_choice

    # for now, just get the one with the most votes and make sure that's the class associated with it
    if image.one_votes + image.zero_votes != 0:
        image.user_score = image.one_votes / (image.one_votes + image.zero_votes)
        # so we can sort by adjusted confidence level based on how sure abnormal vs normal it is
        image.adjusted_user_score = abs(image.user_score - 0.5)
    else:
        image.user_score = 0.5
        image.adjusted_user_score = 0
    image.save()
    return HttpResponseRedirect('/model/' + str(ml_id) + '/numbers-visualization')


def trainNumbersModel(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    t = ModelOperationsNumbers()
    # change from the multithreaded solution since that might be breaking
    t.train_model(ml_model.title)
    return HttpResponseRedirect('/model/' + str(ml_model_id) + '/probability-numbers')


def updateNumbersImagesWithModelPrediction(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    images = ml_model.numberlabel_set.all()
    df = pd.read_csv('final_data_' + ml_model.title + '.csv')
    print("updating numbers images with model prediction")
    for image in images:
        title = "media/ml_model_images/" + ml_model.title + "/" + image.title
        entry = df.loc[df['image'] == title]

        # check for broken image labels or duplicates, needs to be 1
        if len(entry) > 1:
            image.model_classification = entry["probability"].iloc[0]
            image.user_classification = entry["label"].iloc[0]
            image.save()
        else:
            image.model_classification = entry["probability"]
            image.user_classification = entry["label"]
            image.save()
    return HttpResponseRedirect('/model/' + str(ml_model_id) + '/numbers-visualization')

def numbers_model_images_by_model_class(request, ml_model_id, ml_model_classification):
    ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    images = ml_model.numberlabel_set.all().filter(model_classification=int(ml_model_classification))

    return render(request, 'imagelabeling/numbers_model_classification.html',
                  {'images': images, 'ml_model': ml_model, 'ml_model_classification': ml_model_classification})

def numbers_model_images_by_model_class_user_class(request, ml_model_id, ml_model_classification, user_classification):
    ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    images = ml_model.numberlabel_set.all().filter(model_classification=int(ml_model_classification), user_classification=int(user_classification))

    return render(request, 'imagelabeling/numbers_model_classification.html',
                  {'images': images, 'ml_model': ml_model, 'ml_model_classification': ml_model_classification})

def numbers_visualization(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    images = ml_model.numberlabel_set.all()
    images_json = serializers.serialize('json', images)
    # so from each image, we need the adjusted prediction number by the model
    # then the number given by the labelers
    cm_path = './media/final_data_' + ml_model.title + '_probabilities.csv'
    cm = CalculationMultiClass.readCSV(cm_path)
    print(type(cm))
    cm = cm.drop(labels={"dif", "probability"}, axis=1)
    data = cm.to_dict()
    return render(request, 'imagelabeling/number-visualizations-2.html',
                  {'images': images_json, 'ml_model': ml_model, 'cm': data})


def numbers_visualization_2(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningNumbersModel, pk=ml_model_id)
    images = ml_model.numberlabel_set.all()
    images_json = serializers.serialize('json', images)
    # so from each image, we need the adjusted prediction number by the model
    # then the number given by the labelers
    cm_path = './media/final_data_' + ml_model.title + '_probabilities.csv'
    cm = CalculationMultiClass.readCSV(cm_path)
    print(type(cm))
    cm = cm.drop(labels={"dif", "probability"}, axis=1)
    data = cm.to_dict()
    return render(request, 'imagelabeling/number-visualizations-2.html', {'images': images_json, 'ml_model': ml_model, 'cm': data})

# used for testing
def testSkikit(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    t = ModelOperations()
    t.launch_predicting(ml_model.title)

    html = "<html><body>Testing the model against new data!</body></html>"
    return HttpResponse(html)

## all the below if for dynamic modeling which isn't fully implemented yet
def generateAbstractModel(request):
    if request.method == "GET":
        return render(request, 'imagelabeling/create_dynamic_form.html',
                      {'CreateDynamicModelForm': CreateDynamicModelForm})
    else:
        form = CreateDynamicModelForm(request.POST)
        if form.is_valid():
            model_name = request.POST.get("model_name")
            model_name_to_save = ModelSchema.objects.create(name=model_name)

            # should be an int
            number_of_features = request.POST.get("number_of_features")

            #should be an int
            number_of_classifications = request.POST.get("number_of_classifications")

            # create certain number of features
            for x in range(0, int(number_of_features)):
                feature_number = 'feature_' + str(x)
                feature_field_schema = FieldSchema.objects.create(name=feature_number, data_type='character')
                feature = model_name_to_save.add_field(
                    feature_field_schema,
                    null=True,
                    unique=False,
                    max_length=100
                )

            # create certain number of classifications
            for x in range(0, int(number_of_classifications)):
                classification_number = 'classification_' + str(x)
                classification_field_schema = FieldSchema.objects.create(name=classification_number, data_type='character')
                classification = model_name_to_save.add_field(
                    classification_field_schema,
                    null=True,
                    unique=False,
                    max_length=100
                )

            # need to save the dynamic model
            final_model = model_name_to_save.as_model()
            assert issubclass(final_model, models.Model)

            request.session["model_name"] = model_name
            return HttpResponseRedirect('/generate-object-from-model')

def generateObjectFromDynamicModel(request):
    model_name = request.session["model_name"]
    this_model_schema = ModelSchema.objects.get(name=model_name)
    this_model = this_model_schema.as_model()

    instance = this_model.objects.create()
    assert instance.pk is not None
    return HttpResponse("<html><body>" + str(instance.pk) + "</body></html>")

class viewAllModels(ListView):
        model = ModelSchema
        template_name = 'imagelabeling/dynamic_models.html'

def dynamic_model_detail(request, model_id):
    try:
        model = get_object_or_404(ModelSchema, pk=model_id)
        name = model.name
        this_model_schema = ModelSchema.objects.get(name=name)
        fields_object = this_model_schema.get_fields()
        fields = {}
        for field in fields_object:
            this_field = get_object_or_404(FieldSchema, pk=field.id)
            fields[this_field.id] = this_field.name


    except ModelSchema.DoesNotExist:
        raise Http404("Model does not exist")
    return render(request, 'imagelabeling/dynamic_model_detail.html', {'model': model, 'fields': fields})

def viewObjectsOfModel(request):
    model_name = request.session["model_name"]
    test_model_schema = ModelSchema.objects.get(name=model_name)
    model = test_model_schema.as_model()
    objects = model.objects.all()
    html = "<html><body>"

    for object in objects:
        html += str(object.pk)
        html += " "
        print (type(object))

    html += "</body></html>"
    return HttpResponse(html)
