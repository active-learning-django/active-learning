from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404


from .forms import ImageLabelForm, ImageLabelSubmitForm
from .models import ImageLabel

from django.shortcuts import get_object_or_404, render

class HomePageView(ListView):
    model = ImageLabel
    template_name = 'imagelabeling/home.html'

class CreatePostView(CreateView):
    model = ImageLabel
    form_class = ImageLabelSubmitForm
    template_name = 'imagelabeling/post.html'
    success_url = reverse_lazy('home')

def LabelImageView(request):
    model = ImageLabel
    form_class = ImageLabelForm
    # logic to send the image we want user to vote on
    desired_image_set = ImageLabel.objects.order_by('confidence')[:1]
    desired_image = desired_image_set[0]
    context = {
        'desired_image': desired_image,
    }
    template = loader.get_template('imagelabeling/label_image.html')
    success_url = reverse_lazy('home')
    return HttpResponse(template.render(context, request))

def detail(request, image_id):
    try:
        image = ImageLabel.objects.get(pk=image_id)
    except ImageLabel.DoesNotExist:
        raise Http404("Question does not exist")
    return render(request,'imagelabeling/detail.html', {'image': image})

# this will just update our database with the user's vote of whether image is normal or abnormal
# we will envoke this from our form in /label, template label_image
def vote(request, image_id):
    image = get_object_or_404(ImageLabel, pk=image_id)
    choice = request.POST['choice']
    if choice == "Abnormal":
        selected_choice = image.abnormal_votes
        selected_choice += 1
        image.abnormal_votes = selected_choice
    elif choice == "Normal":
        selected_choice = image.normal_votes
        selected_choice += 1
        image.normal_votes = selected_choice
    else:
        selected_choice = image.unknown_votes
        selected_choice += 1
        image.unknown_votes = selected_choice
    # Always return an HttpResponseRedirect after successfully dealing
    # with POST data. This prevents data from being posted twice if a
    # user hits the Back button.
    image.save()
    return HttpResponseRedirect('/label')


