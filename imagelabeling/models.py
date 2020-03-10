from django.db import models

# Create your models here.
from django.db import models

class ImageLabel(models.Model):
    label_option = (
        ("Abnormal", "Abnormal"),
        ("Normal", "Normal"),
        ("Unknown", "Unknown"),
    )

    title = models.CharField(max_length=100)
    category = models.CharField(max_length=100, choices=label_option)
    confidence = models.FloatField(default=0.5)
    adjusted_confidence = models.FloatField(default=0)
    image_file = models.ImageField(upload_to='images/')
    abnormal_votes = models.IntegerField(default=0)
    normal_votes = models.IntegerField(default=0)
    unknown_votes = models.IntegerField(default=0)

    def __str__(self):
        return self.title
