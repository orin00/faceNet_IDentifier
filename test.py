import os
import glob
import torch
import numpy as np
from PIL import Image
import cv2
import json 
from facenet_pytorch import MTCNN, InceptionResnetV1 

BASE_DIR = 'D:/facenet'
TEST_DIR = os.path.join(BASE_DIR, 'test') 
RESULT_DIR = os.path.join(BASE_DIR, 'result') 
MODEL_DB_PATH = os.path.join(BASE_DIR, 'model_db', 'embeddings_db.npz')
LABEL_JSON_PATH = os.path.join(BASE_DIR, 'train_label.json') 
THRESHOLD = 0.9 # 임베딩 거리 임계값 

# device 설정 및 모델 초기화 -> ResNet만 로드
device = torch.device('cpu')
print(f"사용 장치: **{device}** (CUDA nms 오류 우회)")

mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 데이터베이스 변수 -> 전역으로 관리
# 로드된 임베딩
embeddings_db = np.array([])
# 로드된 JSON 라벨 정보
labels_db = {}
# 임베딩 순서와 매칭되는 이름 리스트
labels_db_list = []


# 모델 로드 및 유틸리티
def load_embeddings_db(model_path):
    # 저장된 임베딩 데이터베이스를 로드하는 함수
    global embeddings_db, labels_db_list
    if not os.path.exists(model_path):
        print(f"모델 파일 ({model_path})을 찾을 수 없음")
        return False
    try:
        data = np.load(model_path, allow_pickle=True)
        embeddings_db = data['embeddings']
        labels_db_list = data['names'].tolist()
        print(f"임베딩 데이터베이스 로드 완료: {len(embeddings_db)}개 임베딩")
        return True
    except Exception as e:
        print(f"임베딩 데이터베이스 로드 오류: {e}")
        return False

# JSON 파일에서 인물 라벨 정보를 로드하는 함수 -> train_label.json
def load_labels_from_json(json_path):
    
    global labels_db
    if not os.path.exists(json_path):
        print(f"라벨 JSON 파일 ({json_path})을 찾을 수 없음")
        return False
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labels_db = json.load(f)
        print(f"라벨 JSON 로드 완료: 총 {len(labels_db)}명 정보")
        return True
    except Exception as e:
        print(f"JSON 파일 로드 중 오류: {e}")
        return False

# 임베딩을 비교하여 인물을 식별하고 라벨 정보 반환
def identify_person(query_embedding, embeddings_db, labels_db_list, labels_db_map, threshold=THRESHOLD):
    
    if embeddings_db.size == 0:
        return "Unknown", 0.0, None 

    query_embedding = query_embedding.reshape(1, -1)
    distances = np.linalg.norm(embeddings_db - query_embedding, axis=1)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    if min_dist < threshold:
        name = labels_db_list[min_idx]
        profile_json = labels_db_map.get(name, {}) 
        return name, min_dist, profile_json
    else:
        return "Unknown", min_dist, None

# 새로운 JSON 구조에서 콘솔/저장에 필요한 핵심 정보를 추출하고 재구성
def extract_display_data(profile_json):

    if not profile_json:
        return None
        
    full_name_ko = profile_json.get("full_name_ko", "정보 없음") 
    birth_info = profile_json.get("timeline", {}).get("birth", "정보 없음") 
    job_affiliations = profile_json.get("affiliations", []) 
    main_job = job_affiliations[0] if job_affiliations else "정보 없음"
    
    # 이전 형식에 맞춰 재구성
    return {
        "풀네임": full_name_ko, 
        "영문명": profile_json.get("full_name_en", "정보 없음"), 
        "국적": profile_json.get("nationality", {}).get("country", "정보 없음"), 
        "출생일": birth_info.split(',')[0].strip() if birth_info != "정보 없음" else "정보 없음", 
        "주요직업/소속": main_job, 
        "키워드": profile_json.get("keywords", []) 
    }

# JSON 데이터를 파일로 저장하고 콘솔에 출력하며, 바운딩 박스가 표시된 이미지 저장
def save_and_log_json(result_name, profile_json, log_data, img_cv2_processed):  
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    base_filename = os.path.splitext(result_name)[0]
    json_path = os.path.join(RESULT_DIR, f"RESULT_{base_filename}.json")
    image_path = os.path.join(RESULT_DIR, f"RESULT_{result_name}")
    
    # 식별된 인물이 있을 경우, 출력용 데이터를 재구성
    display_data = extract_display_data(profile_json)
    
    # JSON 파일 저장
    try:
        final_data = {
            "image_file": result_name,
            "recognition_result": log_data,
            # display_data (요약된 정보)를 저장
            "profile_data": display_data,
            # 원본 JSON 데이터도 함께 저장 (선택 사항)
            "raw_profile_data_structure": profile_json 
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"JSON 라벨 정보 저장: {json_path}")
        
    except Exception as e:
        print(f"JSON 파일 저장 오류: {e}")

    # 콘솔에 JSON 로그 출력 (Display Data 위주로 출력)
    print("--- JSON 라벨 정보 출력 (콘솔) ---")
    if display_data:
        # Display Data와 Log Data만 콘솔에 출력
        console_output = {
            "recognition_result": log_data,
            "profile_data": display_data 
        }
        print(json.dumps(console_output, ensure_ascii=False, indent=4))
    else:
         print(json.dumps({"recognition_result": log_data}, ensure_ascii=False, indent=4))
    print("---------------------------------")
    
    # 바운딩 박스가 표시된 이미지 저장
    cv2.imwrite(image_path, img_cv2_processed)
    print(f"{result_name} 결과 이미지 저장 (바운딩 박스 표시): {image_path}")
    
    return image_path


# 테스트 실행부

def run_single_image_test(image_path, test_name):
    # 단일 이미지 파일에서 얼굴을 감지하고 식별하는 함수
    global embeddings_db, labels_db, labels_db_list

    if not os.path.exists(image_path):
        print(f"이미지를 찾을 수 없습니다. 경로: {image_path}")
        return

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"{image_path} 이미지 파일을 읽을 수 없습니다.")
        return
    
    img_cv2_processed = img_cv2.copy() 
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)) 

    # 얼굴 감지
    boxes, probs = mtcnn.detect(img_pil)
    
    print(f"\n--- {test_name} 처리 시작 ---")
    
    if boxes is None or len(boxes) == 0:
        log_data = {"status": "Failure", "detail": "얼굴 감지 실패"}
        save_and_log_json(test_name, None, log_data, img_cv2_processed)
        return

    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    num_faces = len(boxes)
    print(f"얼굴 감지 {num_faces}개")

    # mtcnn.extract은 얼굴 이미지 텐서들을 반환
    face_tensors = mtcnn.extract(img_pil, boxes, save_path=None)
    
    recognition_logs = []
    
    # mtcnn.extract이 단일 4D 텐서 (N, C, H, W)를 반환, 루프를 위해 N개의 3D 텐서 리스트로 변환
    if isinstance(face_tensors, torch.Tensor) and face_tensors.dim() == 4:
         # 4D 텐서를 (1, C, H, W) 텐서 리스트로 분할
         face_tensors = list(torch.split(face_tensors, 1, dim=0))
    elif not isinstance(face_tensors, (list, tuple)):
         # 다른 형태이거나 단일 텐서(3D)일 경우 리스트로 감싸줍니다.
         face_tensors = [face_tensors] if face_tensors is not None else []
    
    
    for i, face_tensor in enumerate(face_tensors):
        
        # 인덱스 안전성 및 None 체크
        # face_tensors 리스트의 길이와 boxes 배열의 길이를 비교하여 루프 안정성 확보
        if i >= num_faces:
            print(f" 얼굴 {i+1}에 해당하는 바운딩 박스 정보(총 {num_faces}개)가 부족하여 루프를 중단합니다.")
            break 
            
        if face_tensor is None:
            print(f" 얼굴 {i+1} 임베딩 추출 실패 (mtcnn.extract 결과가 None). 건너뜁니다.")
            continue

        # 텐서 차원 보정 로직
        # ResNet은 (B, C, H, W) 형태를 예상합니다. (B=1, C=3, H=160, W=160)
        
        # 2D (H, W)인 경우: 채널 차원 (C=1) 추가
        if face_tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            face_tensor = face_tensor.unsqueeze(0) 

        # 3D (C, H, W)인 경우:
        if face_tensor.dim() == 3:
             # 채널이 1개(Grayscale)이면 3채널로 복제
            if face_tensor.shape[0] == 1:
                # (1, H, W) -> (3, H, W)
                face_tensor = face_tensor.expand(3, -1, -1)
            
            # (C, H, W) -> (1, C, H, W) [배치 차원 B=1 추가]
            face_tensor = face_tensor.unsqueeze(0)
            
        # 최종 형태 검증 및 스킵
        if face_tensor.dim() != 4 or face_tensor.shape[1] != 3:
            print(f" 얼굴 {i+1} 임베딩 추출 실패 (최종 형태 {face_tensor.shape}가 (1, 3, 160, 160)이 아님). 스킵합니다.")
            continue
        
        # 임베딩 추출
        with torch.no_grad():
            query_embedding = resnet(face_tensor.to(device)).cpu().numpy().flatten()

        name, min_dist, profile_json = identify_person(
            query_embedding, embeddings_db, labels_db_list, labels_db
        )

        # 바운딩 박스 그리기
        x_min, y_min, x_max, y_max = boxes[i].astype(int)

        # 이미지 색상 재조정
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img_cv2_processed, (x_min, y_min), (x_max, y_max), color, 2)

        log_entry = {
            "face_id": i + 1,
            "box": boxes[i].tolist(),
            "person_id": name,
            "min_distance": f"{min_dist:.3f}"
        }
        recognition_logs.append(log_entry)
        
        if name != "Unknown" and profile_json:
            print(f"[{test_name}] 얼굴 {i+1} - 식별 성공: {name} (Dist: {min_dist:.3f})")
            
            # 식별된 얼굴이 있을 경우 바로 결과 저장 및 종료 -> 단일 인물 이미지라고 가정
            log_data = {"status": "Success", "detail": f"얼굴 {i+1} 식별 완료 ({name})", "faces": recognition_logs}
            save_and_log_json(test_name, profile_json, log_data, img_cv2_processed)
            return
        
        elif name == "Unknown":
            print(f"[{test_name}] 얼굴 {i+1} - 미식별: Unknown (Dist: {min_dist:.3f})")


    # 모든 얼굴 검사 후 식별된 인물 없을 경우
    log_data = {"status": "Partial/Failure", "detail": "식별된 인물 없음", "faces": recognition_logs}
    save_and_log_json(test_name, None, log_data, img_cv2_processed)

# 테스트 폴더의 모든 이미지에 대해 식별을 실행하는 함수
def run_all_test_images():
    
    # .jpg, .jpeg, .png 확장자를 가진 모든 파일을 검색
    test_image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_image_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))

    if not test_image_paths:
        print(f"테스트 폴더 ({TEST_DIR})에 테스트할 이미지 파일이 없습니다.")
        return

    for filepath in sorted(test_image_paths):
        filename = os.path.basename(filepath)
        run_single_image_test(filepath, filename)

# 메인 테스트 코드
if __name__ == '__main__':

    print("데이터베이스 로드 및 초기화")
    
    # 임베딩 데이터베이스 (학습 결과) 로드
    db_loaded = load_embeddings_db(MODEL_DB_PATH)
    
    # JSON 라벨 정보 로드
    json_loaded = load_labels_from_json(LABEL_JSON_PATH)

    if db_loaded and json_loaded:
        print("\n 테스트 이미지 식별 시작")
        run_all_test_images()
    else:
        print("\n 테스트 진행 불가. 임베딩 파일 또는 JSON 파일을 확인하세요.")