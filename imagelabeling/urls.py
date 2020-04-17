from django.urls import path
from . import views

from .views import HomePageView, CreatePostView, viewAllModels


urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('post/', views.CreatePostView, name='add_post'),
    path('model/<int:ml_model_id>/upload/', views.bulk_upload_view, name='upload'),
    path('model/<int:ml_model_id>/test', views.testSkikit, name='test-model'),
    path('model/<int:ml_model_id>/train', views.trainModel, name='train-model'),
    path('model/<int:ml_model_id>/', views.ml_model_detail, name='detail'),
    path('model/<int:ml_model_id>/label/', views.LabelImageView, name='label_image'),
    path('model/<int:ml_model_id>/label/<int:image_id>/', views.image_label_detail, name='detail'),
    path('label/<int:image_id>/vote', views.vote, name='vote'),
    path('model/<int:ml_model_id>/run-predictions', views.updateImagesWithModelPrediction, name='update_images_with_model_prediction'),
    path('model/<int:ml_model_id>/visualization', views.visualization, name='visualization'),
    path('model/<int:ml_model_id>/numbers-visualization', views.numbers_visualization, name='numbers visualization'),
    path('model/<int:ml_model_id>/probability', views.CalculateProbability, name='show roc curve'),
    path('model/<int:ml_model_id>/svm', views.SVMTuning, name='parameter tunning for SVM'),

    path('create-numbers-model', views.CreateNumbersModelView, name='create numbers model'),
    path('model/<int:ml_model_id>/upload-numbers', views.bulk_upload_view_number, name='bulk upload view for numbers'),
    path('model/<int:ml_model_id>/label/<int:number_image_id>/', views.number_image_label_detail, name='detail'),
    path('model/<int:ml_model_id>/train-numbers', views.trainNumbersModel, name='train numbers model'),
    path('model/<int:ml_model_id>/probability-numbers', views.CalculateProbabilityNumbers, name='show roc curve'),
path('model/<int:ml_model_id>/run-predictions-numbers', views.updateNumbersImagesWithModelPrediction, name='update with predictions'),

    path('generate-abstract-model', views.generateAbstractModel, name='generate abstract model'),
    path('view-models', views.viewObjectsOfModel, name='view objects of models'),
    path('view-all-models', viewAllModels.as_view(), name='view models'),
    path('generate-object-from-model', views.generateObjectFromDynamicModel, name='generate object from model'),
    path('dynamic-model/<int:model_id>', views.dynamic_model_detail, name='dynamic model detail'),
]