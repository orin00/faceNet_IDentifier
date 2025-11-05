from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import io
import numpy as np

# 핵심 FaceNet 초기화 및 함수 임포트
from .facenet_init import recognize_face, known_embeddings, names 

class FaceRecognitionAPIView(APIView):

    # POST 요청을 받아 이미지를 식별하고 결과를 JSON으로 반환합니다.
    
    def post(self, request, *args, **kwargs):
        # DB 로드 상태 확인
        if known_embeddings is None or names is None:
            return Response(
                {
                    "result": "error",
                    "message": "FaceNet 데이터베이스가 로드되지 않았거나 초기화에 실패했습니다. 백엔드 로그를 확인하세요."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # 이미지 파일 검증
        if 'image' not in request.FILES:
            return Response(
                {"result": "error", "message": "이미지 파일(key: 'image')이 요청에 포함되지 않았습니다."},
                status=status.HTTP_400_BAD_REQUEST
            )

        uploaded_file = request.FILES['image']
        
        # 파일 내용을 메모리에 읽어 PIL Image 객체로 변환
        try:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            return Response(
                {"result": "error", "message": f"이미지 파일을 읽거나 변환하는 중 오류 발생: {e}"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # 얼굴 인식 함수 호출
        # facenet_init의 recognize_face 함수가 식별 결과 반환
        recognition_result = recognize_face(image)
        
        # 결과 반환
        # recognize_face 함수는 {result, name, distance, threshold} 형태 반환
        
        if recognition_result['result'] in ['success', 'unknown', 'not_found']:
            # 성공, 미식별, 얼굴 미감지 모두 HTTP 200 OK로 처리하고 결과 객체를 반환
            return Response(recognition_result, status=status.HTTP_200_OK)
        else:
            # 내부 모델 오류 발생과 같은 facenet_init에서 정의한 다른 error 케이스 
             return Response(
                recognition_result,
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )