from django.urls import path
from . import views

from .views import HomePageView, CreatePostView, LabelImageView


urlpatterns = [

    path('', HomePageView.as_view(), name='home'),
    path('post/', CreatePostView.as_view(), name='add_post'),
    path('label/', views.LabelImageView, name='label_image')
]