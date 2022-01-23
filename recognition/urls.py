from django.urls import path, include
from recognition import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.home, name='home'),
    path('facecam_feed', views.facecam_feed, name='facecam_feed'),
    # path('success', views.success, name='success'),
    path('detectImage', views.detectImage, name='detectImage'),
    path('train_model', views.Train, name='train_model'),

    ]
