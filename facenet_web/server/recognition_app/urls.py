from django.urls import path
from .views import FaceRecognitionAPIView

urlpatterns = [
    # 이 앱의 URL은 'api/'로 시작
    # 최종 API 경로: http://127.0.0.1:8000/api/recognize/
    path('recognize/', FaceRecognitionAPIView.as_view(), name='recognize'),
]