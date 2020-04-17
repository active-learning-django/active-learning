from django import forms
from .models import ImageLabel
from .models import MachineLearningModel, MachineLearningNumbersModel
from django.utils.translation import gettext_lazy as _
from .models import NumOfIteration


class CreateMachineLearningModelForm(forms.ModelForm):
    class Meta:
        model = MachineLearningModel
        fields = ['title', 'classification_notes']
        labels = {
            'title': _('Title'),
            'classification_notes': _('Classification Notes'),
        }

class CreateMachineLearningNumbersModelForm(forms.ModelForm):
    class Meta:
        model = MachineLearningNumbersModel
        fields = ['title', 'classification_notes']
        labels = {
            'title': _('Title'),
            'classification_notes': _('Classification Notes'),
        }


class ImageLabelForm(forms.ModelForm):
    class Meta:
        model = ImageLabel
        fields = ["image_file", "model_classification"]


class ImageBulkUploadForm(forms.Form):
    bulk_upload = forms.FileField()


# this will generate to the dynamically created model
class CreateDynamicModelForm(forms.Form):
    model_name = forms.CharField(label="model name")
    number_of_classifications = forms.IntegerField(label="number of classifications")
    number_of_features = forms.IntegerField(label="number of features")
    fields = [model_name, "number_of_classifications", "number_of_features"]

class NumOfIterationForm(forms.ModelForm):
    class Meta:
        model = NumOfIteration
        fields = ['Iteration']



class BooleanForm(forms.Form):
    field = forms.TypedChoiceField(coerce=lambda x: x =='True',
                                   choices=((False, 'No'), (True, 'Yes')))


class GammaForm(forms.Form):
    field = forms.CharField(label='Gamma', max_length=100)


kerneloption = (
    (1, "Polynomial"),
    (2, "RBF"),
    (3, "Linear"),
    (4, "Non-Linear"),
)
class SVMKernel(forms.Form):
    field = forms.TypedChoiceField(label="Kernel",coerce=str,
                                   choices=kerneloption)
