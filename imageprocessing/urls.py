from django.urls import path
from . import views


urlpatterns = [
    path('', views.index),
    path('home/', views.index),
    path('about/', views.about),
    path('face_detection/', views.face_detection, name='face_detection'),
    path('process_image/', views.process_image, name='process_image'),
    path('upload/', views.upload, name='upload'),
    path('detect_face/', views.detect_face, name='detect_face'),
    path('feature_matching/', views.feature_matching, name='feature_matching'),
    path('match_feature/', views.match_feature, name='match_feature'),
    path('compare_images/', views.compare_images, name='compare_images'),
    path('compare_images2/', views.compare_images2, name='compare_images2'),
    path('compare_src_dst/', views.compare_src_dst, name='compare_src_dst'),
    path('privacy/', views.privacy, name='privacy'),
    path('terms_and_conditions/', views.terms_and_conditions, name='terms_and_conditions'),
    path('image_filter/', views.image_filter, name='image_filter'),
    path('filter_images/', views.filter_images, name='filter_images'),
    path('success/', views.success, name='success'),


]

