## Active Learning Image Labeling

Django project to implement active labeling

## Database 

This application runs SQLite per the default Django settings

## Image Labeling Application 

### /admin
Admins can upload images (labeled or not labeled) from the admin interface.
These will then be included in the model

### /label
User will be served an image designated by the model. 
The user will record whether the image is normal, abnormal, or unknown.

### Requirements

import scipy
print('scipy: {}'.format(scipy.__version__))
numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))