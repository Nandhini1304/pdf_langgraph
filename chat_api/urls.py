from django.urls import path
from .views import ChatbotView, approve_answer, get_approved_answer

urlpatterns = [
    path('api/chat/', ChatbotView.as_view(), name="chat"),
    path('api/approve/<int:pk>/', approve_answer, name="approve"),
    path('api/answer/<int:pk>/', get_approved_answer, name="get_answer"),
]

