from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy
from django.template import loader
from django.http import HttpResponse


from .forms import ImageLabelForm, ImageLabelSubmitForm
from .models import ImageLabel

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


