
# Create your models here.

# class Label(models.Model):
#     Label_option = (
#         ("Abnormal", "Abnormal"),
#         ("Normal", "Normal"),
#         ("Not Sure", "Not Sure"),
#     )
#     op = models.CharField(max_length=1200, choices= Label_option)
#     # name = models.CharField(max_length=100)
#     pic = models.ImageField(upload_to='images/')
from django.db import models

class Post(models.Model):
    label_option = (
        ("Abnormal", "Abnormal"),
        ("Normal", "Normal"),
        ("Unknown", "Unknown"),
    )

    title = models.CharField(max_length=100, choices=label_option)
    cover = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.title

