from django.db import models

# Create your models here.
from django.db import models
from django.template.defaultfilters import slugify
from django.forms import ModelForm
from django import forms

class MachineLearningModel(models.Model):
    title = models.CharField(max_length=100, unique=True)
    classification_notes = models.CharField(max_length=100)

def get_image_filename(instance, filename):
    title = instance.machine_learning_model.title
    slug_title = slugify(title)
    return "../ml_model_images/%s/%s" % (slug_title, filename)

class ImageLabel(models.Model):
    label_option = (
        ("0", "0"),
        ("1", "1"),
        ("Unknown", "Unknown"),
    )

    title = models.CharField(max_length=100)
    model_classification = models.CharField(max_length=100, choices=label_option)
    confidence = models.FloatField(default=0.5)
    adjusted_confidence = models.FloatField(default=0)
    one_votes = models.IntegerField(default=0)
    zero_votes = models.IntegerField(default=0)
    unknown_votes = models.IntegerField(default=0)
    machine_learning_model = models.ForeignKey(MachineLearningModel, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to=get_image_filename)

    def __str__(self):
        return self.title

class Classifier(models.Model):
    print("classifier model")


class NumOfIteration(models.Model):

    INTEGER_CHOICES = [tuple([x, x]) for x in range(1, 10)]
    Iteration = models.CharField(max_length=100, choices=INTEGER_CHOICES)

    def __str__(self):
        return self.Iteration

