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
from .forms import ImageLabelForm, CreateMachineLearningModelForm, CreateMachineLearningNumbersModelForm, ImageBulkUploadForm, GammaForm, BooleanForm, SVMKernel, CreateDynamicModelForm, DigitForm
from .models import ImageLabel, MachineLearningModel, MachineLearningNumbersModel, ModelSchema, FieldSchema, NumberLabel
from django.shortcuts import get_object_or_404, render

# ml stuff
import pandas as pd

from .ridgemodel import Calculation
from .ridgemodel_multiclass import Calculation as CalculationMultiClass


from pylab import *
import io, urllib, base64
from sklearn.metrics import roc_curve, auc

# get models
from django.apps import apps




class HomePageView(ListView):
    model = MachineLearningModel
    template_name = 'imagelabeling/home.html'

def labelDigitFeatures(request):
    if request.method == "POST":
        form = DigitForm(request.POST)
    else:
        form = DigitForm

    # return HttpResponse("hello")
    return render(request, 'imagelabeling/label_digit.html',{'form':form})

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
def CalculateProbability(request, ml_model_id):
    # get name of model we're running analysis on
    try:
        ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")

    path = 'final_data_test_' + ml_model.title + '.csv'
    #path = '/Users/maggie/Desktop/active-learning/large_data.csv'
    firstdf = Calculation.readCSV(path)
    count = 0

    if request.method == "GET":

            rid_result = Calculation.ridge_regression(firstdf)
            concatedDF = Calculation.concateData(rid_result)
            final = Calculation.ridge_regression(concatedDF)

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

            return HttpResponseRedirect('/model/' + str(ml_model_id) + '/run-predictions')


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

            rid_result = CalculationMultiClass.ridge_regression(firstdf)
            concatedDF = CalculationMultiClass.concateData(rid_result)
            final = CalculationMultiClass.ridge_regression(concatedDF)

            print("       ")
            print(len(final['probability']))
            print("          ")

            CalculationMultiClass.outputCSV(final, ml_model.title)
            CalculationMultiClass.outputJSON(final, ml_model.title)
            return HttpResponseRedirect('/model/' + str(ml_model_id) + '/run-predictions-numbers')

def RenderGraph(request):
    return render(request, 'imagelabeling/graph.html')


# def IterationInputPage(request,ml_model_id):
#
#     if request.method == 'GET':
#         form_class = BooleanForm
#         template_name = 'imagelabeling/temp.html'
#
#         return render(request, 'imagelabeling/temp.html', {'form': form_class})

    # elif request.method == "POST":
        # form_class = NumOfIterationForm
        # input = NumOfIterationForm(request.POST)
        # form_class = BooleanForm
        # input = BooleanForm(request.POST)
        # if input.is_valid():
            # text = input.cleaned_data['Iteration']
            # print(type(text))
            # if input == True:
            # path = '/Users/maggie/Desktop/active-learning/final_data_test.csv'
            #     path = '/Users/maggie/Desktop/active-learning/final_data_test_Test-951.csv'
            #     firstdf = pd.read_csv(path)
            #     firstdf.drop(['Unnamed: 0'], axis=1, inplace=True)
            #     firstdf['dif'] = 0
            #     firstdf['probability'] = 0
            #
            #
            #     tempdf = firstdf
            #     # for i in range(1, text+1):
            #     df = Calculation.ridge_regression(tempdf)
            #     final = Calculation.concateData(df)
            #     #
            #     #show ROC curve
            #     prediction = final['probability']
            #     actual = final['label'].values.reshape(-1, 1)
            #
            #     false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prediction)
            #     roc_auc = auc(false_positive_rate, true_positive_rate)
            #     plt.title('ROC curve for Ridge Regression Model')
            #     plt.subplot(2,1,1)
            #     plt.plot(false_positive_rate, true_positive_rate, 'b',
            #              label='AUC = %0.2f' % roc_auc)
            #     plt.legend(loc='lower right')
            #     plt.plot([0, 1], [0, 1], 'r--')
            #     plt.xlim([-0.1, 1.2])
            #     plt.ylim([-0.1, 1.2])
            #     plt.ylabel('True Positive Rate')
            #     plt.xlabel('False Positive Rate')
            #     fig = plt.gcf()
            #
            #     buf1 = io.BytesIO()
            #     fig.savefig(buf1, format='png')
            #     buf1.seek(0)
            #     string1 = base64.b64encode(buf1.read())
            #     uri1 = urllib.parse.quote(string1)

            # show bar chart
            # plt.subplot(2,1,2)
            # bar_y = tempdf['probability']
            #
            # bar_x = len(tempdf)
            # plt.figure()

            # #color set
            # colors_set = []
            # for value in bar_y:
            #     if 0.2 <= value <= 0.45:
            #         colors_set.append("red")
            #     else:
            #         colors_set.append('green')
            #
            # plt.barh(range(bar_x),bar_y,color = colors_set)
            # # plt.barh(range(bar_x),bar_y)
            #
            # fig = plt.gcf()
            #
            # buf = io.BytesIO()
            # fig.savefig(buf, format='png')
            # buf.seek(0)
            # string = base64.b64encode(buf.read())
            # uri = urllib.parse.quote(string)

            # args = {'data': uri, 'image':uri1}
            #     args = {'image': uri1}
            # return render(request, 'imagelabeling/graph.html', {'data': uri})
            #     return render(request, 'imagelabeling/graph.html', args)


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


def LabelImageView(request, ml_model_id):
    ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    # logic to send the image we want user to vote on
    desired_image_set = ml_model.imagelabel_set.all().order_by('adjusted_user_score')
    if len(desired_image_set) == 0:
        html = "<html><body>Currently no images</body></html>"
        return HttpResponse(html)
    desired_image = desired_image_set[0]
    return render(request, 'imagelabeling/label_image.html', {'desired_image': desired_image, 'ml_model': ml_model})


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
        image.user_score = image.one_votes / (image.one_votes + image.zero_votes)
        # so we can sort by adjusted confidence level based on how sure abnormal vs normal it is
        image.adjusted_user_score = abs(image.user_score - 0.5)
    else:
        image.user_score = 0.5
        image.adjusted_user_score = 0
    image.save()
    return HttpResponseRedirect('/model/' + str(ml_id) + '/label')

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
