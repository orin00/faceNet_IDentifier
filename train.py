import os
import glob
import torch
import numpy as np
from PIL import Image
import json
from facenet_pytorch import MTCNN, InceptionResnetV1 

# 설정 및 초기화
BASE_DIR = 'D:/facenet'
ENROLL_DIR = BASE_DIR 
MODEL_DB_PATH = os.path.join(BASE_DIR, 'model_db', 'embeddings_db.npz')
LABEL_JSON_PATH = os.path.join(BASE_DIR, 'train_label.json') 

# device 설정 및 모델 초기화
device = torch.device('cpu') 
print(f"사용 장치: **{device}** (CUDA nms 오류 우회)")

mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 유틸리티 함수
def load_labels_from_json(json_path):
    """JSON 파일에서 인물 라벨 정보를 로드하는 함수"""
    if not os.path.exists(json_path):
        print(f"경고: 라벨 JSON 파일 ({json_path})을 찾을 수 없습니다.")
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labels_db = json.load(f)
        print(f"라벨 데이터베이스 로드 완료: 총 {len(labels_db)}명 정보")
        return labels_db
    except Exception as e:
        print(f"JSON 파일 로드 중 오류: {e}")
        return {}

# 임베딩 데이터베이스를 NumPy 파일로 저장하는 함수
def save_embeddings_db(embeddings_db, labels_db_list, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        np.savez_compressed(
            model_path, 
            embeddings=embeddings_db, 
            names=np.array(labels_db_list)
        )
        print(f"\n 임베딩 데이터베이스 저장 성공: {model_path}")
    except Exception as e:
        print(f" 임베딩 데이터베이스 저장 실패: {e}")


# 메인 학습 코드
def run_training():
    print("1. 인물 데이터베이스 구축 시작 (임베딩 추출)")

    # 라벨 정보 로드
    labels_db = load_labels_from_json(LABEL_JSON_PATH) 

    if not labels_db:
        print("라벨 데이터베이스가 비어있어 임베딩 추출을 시작할 수 없습니다.")
        return

    embeddings = []
    names = []

    if not os.path.exists(ENROLL_DIR):
        print(f"등록 폴더 ({ENROLL_DIR})를 찾을 수 없습니다.")
        return

    # 'model_db', 'result', 'test' 등은 인물 폴더가 아니므로 제외
    EXCLUDE_DIRS = ['model_db', 'result', 'test'] 
    
    # ENROLL_DIR 내의 모든 폴더를 순회하며 인물 폴더를 찾음
    for name in sorted(os.listdir(ENROLL_DIR)):
        person_id = name # 폴더 이름을 인물 ID로 사용
        person_dir = os.path.join(ENROLL_DIR, person_id)

        # 폴더가 아니거나, 숨겨진 폴더이거나, 제외 목록에 있으면 스킵
        if not os.path.isdir(person_dir) or person_id.startswith('.') or person_id.lower() in EXCLUDE_DIRS:
            continue
        
        # JSON 라벨에 인물 ID가 없으면 경고 후 스킵
        if person_id not in labels_db:
             print(f"인물 폴더 '{person_id}'에 대한 JSON 라벨 정보가 없습니다. 스킵합니다.")
             continue
        
        # 인물 폴더 내부에서 이미지 검색 [JPG, JPEG, PNG]
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(person_dir, ext)))

        if not image_paths:
            print(f"폴더 '{person_id}'에서 등록용 이미지 파일을 찾을 수 없습니다. 스킵합니다.")
            continue

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            try:
                # 얼굴 감지 및 정렬
                img_pil = Image.open(image_path).convert('RGB')
                face_tensor = mtcnn(img_pil, return_prob=False)

                if face_tensor is None:
                    # 얼굴 감지 실패
                    continue

                if face_tensor.dim() == 3: face_tensor = face_tensor.unsqueeze(0)

                # 임베딩 추출
                with torch.no_grad():
                    embedding = resnet(face_tensor.to(device)).cpu().numpy().flatten()

                embeddings.append(embedding)
                names.append(person_id) # 폴더 이름(ID)을 라벨로 사용
                print(f"{person_id} ({filename}) 임베딩 추출 완료")

            except Exception as e:
                print(f"  - 오류 발생 ({filename}): {e}")

    # 데이터베이스 저장
    if embeddings:
        embeddings_db = np.array(embeddings)
        print(f"\n임베딩 추출 완료: 총 {len(embeddings)}개 임베딩 등록, {len(set(names))}명")
        save_embeddings_db(embeddings_db, names, MODEL_DB_PATH)
    else:
        print("\n임베딩 추출에 실패했습니다. (등록된 임베딩이 0개인 경우)")


if __name__ == '__main__':
    run_training()