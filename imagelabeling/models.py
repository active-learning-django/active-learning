from django.db import models

# Create your models here.
from django.db import models

class MachineLearningModel(models.Model):
    title = models.CharField(max_length=100)

class ImageLabel(models.Model):
    label_option = (
        ("Abnormal", "Abnormal"),
        ("Normal", "Normal"),
        ("Unknown", "Unknown"),
    )

    title = models.CharField(max_length=100)
    model_classification = models.CharField(max_length=100, choices=label_option)
    confidence = models.FloatField(default=0.5)
    adjusted_confidence = models.FloatField(default=0)
    image_file = models.ImageField(upload_to='images/')
    abnormal_votes = models.IntegerField(default=0)
    normal_votes = models.IntegerField(default=0)
    unknown_votes = models.IntegerField(default=0)
    machine_learning_model = models.ForeignKey(MachineLearningModel, on_delete=models.CASCADE)

    def __str__(self):
        return self.title

class Classifier(models.Model):
    print("classifier model")
