from django import forms
from .models import ImageLabel
from .models import MachineLearningModel
from django.utils.translation import gettext_lazy as _

class CreateMachineLearningModelForm(forms.ModelForm):
    class Meta:
        model = MachineLearningModel
        fields = ['title', 'classification_notes']
        labels = {
            'title': _('Title'),
            'classification_notes': _('Classification Notes'),
        }

class ImageLabelForm(forms.ModelForm):
    class Meta:
        model = ImageLabel
        fields = ["image_file", "model_classification"]

