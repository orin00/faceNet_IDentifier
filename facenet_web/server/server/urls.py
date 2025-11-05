from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # recognition_app의 URL을 'api/' 경로에 연결
    # 최종 경로는 http://127.0.0.1:8000/api/recognize/
    path('api/', include('recognition_app.urls')),
]

# 개발 환경에서 Media 파일(업로드된 이미지)을 서빙하기 위한 설정 추가
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)