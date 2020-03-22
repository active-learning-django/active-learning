from django import forms
from .models import ImageLabel
from .models import MachineLearningModel

class CreateMachineLearningModelForm(forms.ModelForm):
    class Meta:
        model = MachineLearningModel
        fields = ['title']

class ImageLabelForm(forms.ModelForm):
    class Meta:
        model = ImageLabel
        fields = ["image_file"]

