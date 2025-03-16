from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_view, name='main'),
    path('region/', views.stream_video_counting_region, name='stream_region'),
    path('barplot/',views.get_bar_plot,name='chart1'),
    path('lineplot/',views.get_line_plot,name='chart2')
    
]
