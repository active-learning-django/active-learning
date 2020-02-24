from django.db import models

# Create your models here.
from django.db import models

class ImageLabel(models.Model):
    # ...
    def __str__(self):
        return self.imagelabel_text