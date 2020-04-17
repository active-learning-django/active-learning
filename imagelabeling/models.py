
# Create your models here.
from django.db import models
from django.template.defaultfilters import slugify
from django.forms import ModelForm
from django import forms
<<<<<<< HEAD

# dynamic models
from dynamic_models.models import AbstractModelSchema, AbstractFieldSchema


class ModelSchema(AbstractModelSchema):
    pass


class FieldSchema(AbstractFieldSchema):
    pass



=======
>>>>>>> 097c50d8f457df51a2502765030ae222ff46ee18

class MachineLearningModel(models.Model):
    title = models.CharField(max_length=100, unique=True)
    classification_notes = models.CharField(max_length=100)


def get_image_filename(instance, filename):
    title = instance.machine_learning_model.title
    slug_title = slugify(title)
    return "../ml_model_images/%s/%s" % (slug_title, filename)
<<<<<<< HEAD

=======
>>>>>>> 097c50d8f457df51a2502765030ae222ff46ee18

class ImageLabel(models.Model):
    label_option = (
        ("0", "0"),
        ("1", "1"),
        ("Unknown", "Unknown"),
    )

    title = models.CharField(max_length=100)
    model_classification = models.CharField(max_length=100, choices=label_option)
    user_score = models.FloatField(default=0.5)
    adjusted_user_score = models.FloatField(default=0)
    model_score = models.FloatField(default=0)
<<<<<<< HEAD
    model_difference = models.FloatField(default=0)
    model_probability = models.FloatField(default=0)
=======
>>>>>>> 097c50d8f457df51a2502765030ae222ff46ee18
    one_votes = models.IntegerField(default=0)
    zero_votes = models.IntegerField(default=0)
    unknown_votes = models.IntegerField(default=0)
    machine_learning_model = models.ForeignKey(MachineLearningModel, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to=get_image_filename)

    def __str__(self):
        return self.title


class NumberLabel(models.Model):
    label_option = (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        ("Unknown", "Unknown"),
    )

    title = models.CharField(max_length=100)
    user_classification = models.CharField(max_length=100, choices=label_option)
    model_classification = models.CharField(max_length=100, choices=label_option)
    user_score = models.FloatField(default=0.5)
    adjusted_user_score = models.FloatField(default=0)
    model_score = models.FloatField(default=0)
    model_difference = models.FloatField(default=0)
    model_probability = models.FloatField(default=0)
    zero_votes = models.IntegerField(default=0)
    one_votes = models.IntegerField(default=0)
    two_votes = models.IntegerField(default=0)
    three_votes = models.IntegerField(default=0)
    four_votes = models.IntegerField(default=0)
    five_votes = models.IntegerField(default=0)
    six_votes = models.IntegerField(default=0)
    seven_votes = models.IntegerField(default=0)
    eight_votes = models.IntegerField(default=0)
    nine_votes = models.IntegerField(default=0)
    unknown_votes = models.IntegerField(default=0)
    machine_learning_model = models.ForeignKey(MachineLearningModel, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to=get_image_filename)


class Classifier(models.Model):
    print("classifier model")


class NumOfIteration(models.Model):

    INTEGER_CHOICES = [tuple([x, x]) for x in range(1, 10)]
<<<<<<< HEAD
    Iteration = models.IntegerField(max_length=100, choices=INTEGER_CHOICES)
=======
    Iteration = models.CharField(max_length=100, choices=INTEGER_CHOICES)
>>>>>>> 097c50d8f457df51a2502765030ae222ff46ee18

    def __str__(self):
        return self.Iteration

