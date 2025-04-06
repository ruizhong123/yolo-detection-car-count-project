from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_view, name='main_view'),
    path('stream/', views.stream_video, name='stream_video'),
    
    path('data/', views.get_chart_data, name='get_data')
]