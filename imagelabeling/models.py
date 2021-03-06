
# Create your models here.
from django.db import models
from django.template.defaultfilters import slugify
from django.forms import ModelForm
from django import forms

# dynamic models
from dynamic_models.models import AbstractModelSchema, AbstractFieldSchema


class ModelSchema(AbstractModelSchema):
    pass


class FieldSchema(AbstractFieldSchema):
    pass




class MachineLearningModel(models.Model):
    title = models.CharField(max_length=100, unique=True)
    classification_notes = models.CharField(max_length=100)

class MachineLearningNumbersModel(models.Model):
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
    user_score = models.FloatField(default=0.5)
    adjusted_user_score = models.FloatField(default=0)
    model_score = models.FloatField(default=0)
    model_difference = models.FloatField(default=0)
    model_probability = models.FloatField(default=0)
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

    shapes = (
        (0, "Oval"),
        (1, "Stick"),
        (2, "Swan"),
        (3, "Butterfly"),
        (4, "Flag"),
        (5, "Hook"),
        (6, "Combination Clock"),
        (7, "Boomerang"),
        (8, "Snowman"),
        (9, "Balloon on a String"),
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
    horizontal_line = models.IntegerField(default=0)
    vertical_line = models.IntegerField(default=0)
    loops = models.IntegerField(default=0)
    close_eye_hook = models.IntegerField(default=0)
    open_eye_hook = models.IntegerField(default=0)
    acute_angle = models.IntegerField(default=0)
    right_angle = models.IntegerField(default=0)
    curves = models.IntegerField(default=0)
    shape_options = models.CharField(max_length=100, choices=shapes)
    machine_learning_model = models.ForeignKey(MachineLearningNumbersModel, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to=get_image_filename)


class Classifier(models.Model):
    print("classifier model")


class NumOfIteration(models.Model):

    INTEGER_CHOICES = [tuple([x, x]) for x in range(1, 10)]
    Iteration = models.IntegerField(max_length=100, choices=INTEGER_CHOICES)

    def __str__(self):
        return self.Iteration

# class UserInfo(models.Model):
#     username = models.CharField(max_length=32, verbose_name="用户名")
#     password = models.CharField(max_length=32, verbose_name="密码")
#     nickname = models.CharField(max_length=32, verbose_name="姓名")
#     phone = models.CharField(max_length=11, verbose_name="电话")
#     email = models.EmailField(verbose_name="邮箱")

class DigitFeature(models.Model):
    total_digit = models.IntegerField(null=True)
    horizontal_line = models.IntegerField(null=True)
    vertical_line = models.IntegerField(null=True)
    loops = models.IntegerField(null=True)
    close_eye_hook = models.IntegerField(null=True)
    open_eye_hook = models.IntegerField(null=True)
    acute_Angle = models.IntegerField(null=True)
    right_Angle = models.IntegerField(null=True)
    # curve = models.IntegerField(null=True)
    label = models.IntegerField(null=True)
    # machine_learning_model = models.ForeignKey(MachineLearningNumbersModel, on_delete=models.CASCADE)
    # image_file = models.ImageField(upload_to=get_image_filename)
    #
    # def __str__(self):
    #     list = [self.total_digit, self.horizontal_line, self.vertical_line, self.loops,self.close_eye_hook,self.open_eye_hook,self.acute_Angle,self.right_Angle, self.curve]
    #     return list
#
class AlphaInput(models.Model):
    alpha_input = models.FloatField(default=0)

    def __float__(self):
        return self.alpha_input

class ModelEvaluation(models.Model):
    alpha_value = models.FloatField(default=0)
    r_score = models.FloatField(default=0)
    auc = models.FloatField(default=0)


