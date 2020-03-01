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
pip install -r requirements.txt