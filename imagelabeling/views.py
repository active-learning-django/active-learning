from django.shortcuts import render
from django.http import HttpResponse
from .test_skikit import Test_Skikit
from .model_operations import ModelOperations

# Create your views here.
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404
from django.forms import modelformset_factory

from .forms import ImageLabelForm, CreateMachineLearningModelForm, ImageBulkUploadForm
from .models import ImageLabel, MachineLearningModel
from django.core.files import File

from zipfile import *

from django.shortcuts import get_object_or_404, render


class HomePageView(ListView):
    model = MachineLearningModel
    template_name = 'imagelabeling/home.html'


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
            print("name " + name)
            with zip_file.open(name) as f:
                image_file = f.read()

                print("name again")
                with open(name + '.jpg', 'wb+') as destination:
                    destination.write(image_file)
                    fileName = File(open(name + '.jpg', "wb+"))
                    photo = ImageLabel(machine_learning_model=model, image_file=fileName, title=name)
                    print("about to save")
                    photo.save()


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




