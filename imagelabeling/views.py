from django.shortcuts import render
from django.http import HttpResponse
from .test_skikit import Test_Skikit

# Create your views here.
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404
from django.forms import modelformset_factory

from .forms import ImageLabelForm, CreateMachineLearningModelForm
from .models import ImageLabel, MachineLearningModel



from django.shortcuts import get_object_or_404, render

class HomePageView(ListView):
    model = ImageLabel
    template_name = 'imagelabeling/home.html'

def CreatePostView(request):
    model = MachineLearningModel
    form_class = CreateMachineLearningModelForm
    template_name = 'imagelabeling/post.html'
    success_url = reverse_lazy('home')
    ImageFormSet = modelformset_factory(ImageLabel,
                                        form=ImageLabelForm, extra=10)
    if request.method == 'POST':

        createMLModelForm = CreateMachineLearningModelForm(request.POST)
        formset = ImageFormSet(request.POST, request.FILES,
                               queryset=ImageLabel.objects.none())

        if createMLModelForm.is_valid() and formset.is_valid():
            create_model_form = createMLModelForm.save(commit=False)
            create_model_form.title = request.POST.get('title')
            create_model_form.save()

            # for each image label form
            for form in formset.cleaned_data:
                # this helps to not crash if the user
                # does not upload all the photos
                # save the image objects to database
                if form:
                    image_file = form['image_file']
                    # print(request.FILES)
                    # sets up the foreign key association
                    photo = ImageLabel(machine_learning_model=create_model_form, image_file=image_file, title=image_file)
                    photo.save()
            # after that's done, we can train the model
            # this might not be right
            ml_model = get_object_or_404(MachineLearningModel, pk=create_model_form.id)
            t = Test_Skikit()
            t.launch_training(ml_model.title)
            html = "<html><body>Training your model!</body></html>"
            return HttpResponse(html)
            # redirect to train the model
        else:
            print(createMLModelForm.errors, formset.errors)
    else:
        createMLModelForm = CreateMachineLearningModelForm()
        formset = ImageFormSet(queryset=ImageLabel.objects.none())
    return render(request, 'imagelabeling/post.html',
                  {'CreateMachineLearningModelForm': CreateMachineLearningModelForm, 'formset': formset})

def ml_model_detail(request, ml_model_id):
    try:
        ml_model = MachineLearningModel.objects.get(pk=ml_model_id)
    except MachineLearningModel.DoesNotExist:
        raise Http404("Model does not exist")
    return render(request, 'imagelabeling/model_detail.html', {'ml_model': ml_model})

def LabelImageView(request):
    model = ImageLabel
    form_class = ImageLabelForm
    # logic to send the image we want user to vote on
    desired_image_set = ImageLabel.objects.order_by('adjusted_confidence')
    if len(desired_image_set) == 0:
        html = "<html><body>Currently no images</body></html>"
        return HttpResponse(html)
    desired_image = desired_image_set[0]
    context = {
        'desired_image': desired_image,
    }
    template = loader.get_template('imagelabeling/label_image.html')
    success_url = reverse_lazy('home')
    return HttpResponse(template.render(context, request))

def image_label_detail(request, image_id):
    try:
        image = ImageLabel.objects.get(pk=image_id)
    except ImageLabel.DoesNotExist:
        raise Http404("Label does not exist")
    return render(request, 'imagelabeling/image_label_detail.html', {'image': image})

# this will just update our database with the user's vote of whether image is normal or abnormal
# we will envoke this from our form in /label, template label_image
def vote(request, image_id):
    image = get_object_or_404(ImageLabel, pk=image_id)
    choice = request.POST['choice']
    if choice == 1:
        selected_choice = image.one_votes
        selected_choice += 1
        image.one_votes = selected_choice
    elif choice == 0:
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
    return HttpResponseRedirect('/label')

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




