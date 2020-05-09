from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views


urlpatterns = [
    path('', views.index),
    path('home/', views.index),
    path('about/', views.about),
    path('contact', views.contact, name='contact'),
    path('face_detection/', views.face_detection),
    path('process_image/', views.process_image),
    path('upload/', views.upload),
    path('detect_face/', views.detect_face, name='detect_face')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)