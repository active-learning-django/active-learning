from django.shortcuts import render
from django.http import HttpResponse
import csv
from .test_skikit import Test_Skikit
from .model_operations import ModelOperations

# views here.
from django.views.generic import ListView, CreateView
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404

# for turning queryset to json
from django.core import serializers

# zipfile for handling bulk upload
from zipfile import *

# get the forms
from .forms import ImageLabelForm, CreateMachineLearningModelForm, ImageBulkUploadForm, NumOfIterationForm, BooleanForm
from .models import ImageLabel, MachineLearningModel, NumOfIteration
from django.shortcuts import get_object_or_404, render

# ml stuff
import pandas as pd


from .ridgemodel import Calculation

from pylab import *
import io, urllib, base64
from sklearn.metrics import roc_curve, auc




class HomePageView(ListView):
    model = MachineLearningModel
    template_name = 'imagelabeling/home.html'



def IterationInputPage(request,ml_model_id):
    path = '/Users/maggie/Desktop/active-learning/large_data.csv'
    firstdf = Calculation.readCSV(path)
    count = 0

    if request.method == "GET":

            rid_result = Calculation.ridge_regression(firstdf)
            concatedDF = Calculation.concateData(rid_result)
            final = Calculation.ridge_regression(concatedDF)

            print("       ")
            print(len(final['probability']))
            print("          ")

            Calculation.outputCSV(final)
            Calculation.outputJSON(final)


            plt = Calculation.ROC(final)
            fig = plt.gcf()

            buf1 = io.BytesIO()
            fig.savefig(buf1, format='png')
            buf1.seek(0)
            string1 = base64.b64encode(buf1.read())
            uri1 = urllib.parse.quote(string1)

            form_class = BooleanForm
            args = {'image': uri1, 'form': form_class}

            return render(request, 'imagelabeling/graph.html', args)


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


# this trains the model once, then should redirect to iteration page i think
def trainModel(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    t = ModelOperations()
    # change from the multithreaded solution since that might be breaking
    t.train_model(ml_model.title)
    return HttpResponseRedirect('/model/' + str(ml_model_id) + '/run-predictions')


def updateImagesWithModelPrediction(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    images = ml_model.imagelabel_set.all()
    df = pd.read_csv('final_data_test_' + ml_model.title + '.csv')
    for image in images:
        title = "media/ml_model_images/" + ml_model.title + "/" + image.title
        entry = df.loc[df['image'] == title]
        print(entry)
        # check for broken image labels
        if len(entry) > 0:
            image.model_score = entry["label"]
            image.save()
    return HttpResponseRedirect('/model/' + str(ml_model_id) + '/prob')


# want to get all the image labels for this model
# then we want to compare how the users labeled the model,
# to how the model predicts these labels
# so we can double check them, pretty much
# def visualization(request, ml_model_id):
#     ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
#     images = ml_model.imagelabel_set.all()
#     images_json = serializers.serialize('json', images)
#     # so from each image, we need the adjusted prediction number by the model
#     # then the number given by the labelers
#     html = "<html><body>Visualization will go here!</body></html>"
#     return render(request, 'imagelabeling/visualizations.html', {'images': images_json, 'ml_model': ml_model})

def visualization(request, ml_model_id):
    html = "<html><body>Visualization will go here!</body></html>"
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    images = ml_model.imagelabel_set.all()
    images_json = serializers.serialize('json', images)

    df = pd.read_csv("/Users/maggie/Desktop/active-learning/output.csv")
    df = df[df.columns[-4:]]
    df = df.to_json(orient="index")
    # df1 = df['probability']
    # df = pd.read_json("/Users/maggie/Desktop/active-learning/output.json")
    # df_json = df.to_json(df)
    # template = loader.get_template('imagelabeling/visual2.html')

    # return HttpResponse(template.render())
    return render(request, 'imagelabeling/visual2.html', {"image": df ,'images': images_json, 'ml_model': ml_model})





def testSkikit(request, ml_model_id):
    ml_model = get_object_or_404(MachineLearningModel, pk=ml_model_id)
    t = ModelOperations()
    t.launch_predicting(ml_model.title)

    html = "<html><body>Testing the model against new data!</body></html>"
    return HttpResponse(html)


