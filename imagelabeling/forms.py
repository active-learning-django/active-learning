from django import forms
from .models import ImageLabel

class ImageLabelSubmitForm(forms.ModelForm):
    class Meta:
        model = ImageLabel
        fields = ['title', 'model_classification', 'image_file']

class ImageLabelForm(forms.ModelForm):
    Choices = forms.ChoiceField(choices=[(1, 'abnormal_votes'), (1, 'normal_votes'), (1, 'unknown_votes')])
    class Meta:
        model = ImageLabel
        fields = ["Choices"]

