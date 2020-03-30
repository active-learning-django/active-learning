from django.urls import path
from . import views

from .views import HomePageView, CreatePostView


urlpatterns = [

    path('', HomePageView.as_view(), name='home'),
    path('post/', views.CreatePostView, name='add_post'),
    path('upload/', views.bulk_upload_view, name='upload'),
    path('test-model/', views.testSkikit, name='test-model'),
    path('train-model/', views.trainModel, name='train-model'),
    path('model/<int:ml_model_id>/', views.ml_model_detail, name='detail'),
    path('model/<int:ml_model_id>/label/', views.LabelImageView, name='label_image'),
    path('label/<int:image_id>/', views.image_label_detail, name='detail'),
    path('label/<int:image_id>/vote/', views.vote, name='vote'),
]