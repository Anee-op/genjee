 

from django.urls import path 
from . import views 

urlpatterns = [
    path('', views.home_view, name='home'), 
    path('qa/<str:college_slug>/', views.college_qa_view, name='college_qa'),
]
