import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path
from PIL import Image
import json

from django.conf import settings

# 설정 및 초기화
MODEL_DB_PATH = settings.EMBEDDINGS_DB_PATH
LABEL_JSON_PATH = settings.LABEL_JSON_PATH

# 유클리드 임계값
THRESHOLD = 1.0 

# device 설정 및 모델 초기화
device = torch.device('cpu') 
print(f"FaceNet 초기화 - 사용 장치: {device}")

try:
    mtcnn = MTCNN(image_size=160, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
except Exception as e:
    print(f"모델 초기화 오류: {e}")
    mtcnn = None
    resnet = None

# 데이터베이스 변수
known_embeddings = None
names = None
labels_db = {} # 프로필 정보를 담을 딕셔너리 추가


# 임베딩 및 라벨 데이터 로드
def load_labels_db():
    # JSON 파일에서 인물 라벨 정보를 로드하는 함수
    global labels_db
    if not os.path.exists(LABEL_JSON_PATH):
        print(f"경고: 라벨 JSON 파일 '{LABEL_JSON_PATH}'를 찾을 수 없습니다.")
        return False
    try:
        with open(LABEL_JSON_PATH, 'r', encoding='utf-8') as f:
            labels_db = json.load(f)
        print(f"라벨 JSON 로드 완료: 총 {len(labels_db)}명 정보")
        return True
    except Exception as e:
        print(f"JSON 파일 로드 중 오류: {e}")
        return False

if mtcnn and resnet:
    # 임베딩 데이터 로드
    try:
        data = np.load(MODEL_DB_PATH)
        # 텐서를 CPU로 로드
        known_embeddings = torch.tensor(data['embeddings'], device=device)
        # np.bytes_ 처리 없이 names 로드
        names = data['names'].tolist() 
        print(f"FaceNet DB 로드 성공. 등록된 임베딩 수: {len(names)}")
    except FileNotFoundError:
        print(f"경고: 모델 DB 파일 '{MODEL_DB_PATH}'를 찾을 수 없습니다. 인식 기능을 사용할 수 없습니다.")
    except Exception as e:
        print(f"모델 DB 로드 중 오류 발생: {e}")

    # 라벨 JSON 로드
    load_labels_db()

# 유틸리티 함수

def extract_display_data(profile_json):
    """
    JSON 구조에서 콘솔/저장에 필요한 핵심 정보를 추출하고 재구성합니다.
    """
    if not profile_json:
        return None
        
    full_name_ko = profile_json.get("full_name_ko", "정보 없음") 
    birth_info = profile_json.get("timeline", {}).get("birth", "정보 없음") 
    job_affiliations = profile_json.get("affiliations", []) 
    main_job = job_affiliations[0] if job_affiliations else "정보 없음"
    
    # 웹 클라이언트에 전달할 요약 정보
    return {
        "풀네임": full_name_ko, 
        "영문명": profile_json.get("full_name_en", "정보 없음"), 
        "국적": profile_json.get("nationality", {}).get("country", "정보 없음"), 
        "출생일": birth_info.split(',')[0].strip() if birth_info != "정보 없음" else "정보 없음", 
        "주요직업/소속": main_job, 
        "키워드": profile_json.get("keywords", []) 
    }


# 얼굴 인식 함수 정의
def recognize_face(image: Image.Image):

    # PIL Image 객체를 입력받아 얼굴을 식별하고 이름과 프로필 정보 반환
    if known_embeddings is None or names is None or not labels_db:
        # DB 로드가 안 되었을 경우의 오류 처리, views.py함께 이중 체크
        return {"result": "error", "message": "FaceNet 데이터베이스(임베딩 또는 라벨)가 로드되지 않았습니다."}
    
    # 얼굴 감지 및 정렬
    face_tensor = mtcnn(image)
    
    if face_tensor is None:
        return {"result": "not_found", "message": "이미지에서 얼굴을 찾을 수 없습니다."}
    
    # 임베딩 벡터 생성 (단일 얼굴 가정)
    if face_tensor.ndim == 3: face_tensor = face_tensor.unsqueeze(0)
    
    # 텐서 차원 검증
    if face_tensor.dim() != 4 or face_tensor.shape[1] != 3:
         return {"result": "error", "message": f"얼굴 이미지 텐서 형태 오류: {face_tensor.shape}"}
    
    face_tensor = face_tensor.to(device)
    
    with torch.no_grad():
        face_embedding_tensor = resnet(face_tensor).detach().cpu()
        face_embedding_np = face_embedding_tensor[0].numpy() # 512차원 벡터

    # 데이터베이스와 비교
    known_embeddings_np = known_embeddings.cpu().numpy()
    
    # 유클리드 거리 (L2 norm) 계산
    distances = np.linalg.norm(known_embeddings_np - face_embedding_np, axis=1)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    
    best_match_name = names[min_idx]
    
    
    if min_dist < THRESHOLD:
        # 식별 성공: 이름과 프로필 정보를 함께 반환
        profile_json_raw = labels_db.get(best_match_name, {})
        profile_json_summary = extract_display_data(profile_json_raw)

        return {
            "result": "success",
            "name": best_match_name,
            "distance": round(float(min_dist), 4),
            "threshold": THRESHOLD,
            "profile_data": profile_json_summary, # 요약된 프로필 데이터 추가
            "raw_profile_data": profile_json_raw # 원본 프로필 데이터 추가
        }
    else:
        # 미식별
        return {
            "result": "unknown",
            "name": "Unknown",
            "distance": round(float(min_dist), 4),
            "threshold": THRESHOLD
        }