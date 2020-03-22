from django.contrib import admin

# Register your models here.
from django.contrib import admin

from .models import MachineLearningModel
from .models import ImageLabel


admin.site.register(ImageLabel)
admin.site.register(MachineLearningModel)