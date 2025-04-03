from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_view, name='main_view'),
    path('stream/', views.stream_video_counting_region, name='stream_video_counting_region'),
    path('charts/data/', views.get_chart_data, name='get_chart_data'),
]