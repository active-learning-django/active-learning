from django.urls import path
from . import views

from .views import HomePageView, CreatePostView


urlpatterns = [

    path('', HomePageView.as_view(), name='home'),
    path('post/', CreatePostView.as_view(), name='add_post'),
    path('label/', views.LabelImageView, name='label_image'),
    path('test-model/', views.testSkikit, name='test-model'),
    path('<int:image_id>/', views.detail, name='detail'),
    path('<int:image_id>/vote/', views.vote, name='vote'),
]