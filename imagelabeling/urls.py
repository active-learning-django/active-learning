from django.urls import path
from . import views

from .views import HomePageView, CreatePostView


urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('post/', views.CreatePostView, name='add_post'),
    path('model/<int:ml_model_id>/upload/', views.bulk_upload_view, name='upload'),
    path('model/<int:ml_model_id>/test', views.testSkikit, name='test-model'),
    path('model/<int:ml_model_id>/train', views.trainModel, name='train-model'),
    path('model/<int:ml_model_id>/', views.ml_model_detail, name='detail'),
    path('model/<int:ml_model_id>/label/', views.LabelImageView, name='label_image'),
    path('it/', views.IterationInputPage, name='ask_user_for_num_interation'),
    path('model/<int:ml_model_id>/label/<int:image_id>/', views.image_label_detail, name='detail'),
    path('/label/<int:image_id>/vote/', views.vote, name='vote'),
    path('model/<int:ml_model_id>/prob/', views.DisplayROC, name='display_ROC_Curve'),
    path('model/<int:ml_model_id>/run-predictions/', views.updateImagesWithModelPrediction, name='update_images_with_model_prediction'),
    path('model/<int:ml_model_id>/visualization/', views.visualization, name='visualization'),

]