from django.urls import path
from . import views

from .views import HomePageView, CreatePostView


urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('post/', views.CreatePostView, name='add_post'),
    path('label/', views.LabelImageView, name='label_image'),
    path('test-model/', views.testSkikit, name='test-model'),
    path('train-model/', views.trainModel, name='train-model'),
    path('model/<int:ml_model_id>/', views.ml_model_detail, name='detail'),
    path('label/<int:image_id>/', views.image_label_detail, name='detail'),
    path('label/<int:image_id>/vote/', views.vote, name='vote'),
    path('iteration/', views.IterationInputPage, name='ask_user_for_num_interation'),
    path('prob/', views.DisplayROC, name='display_ROC_Curve'),
]