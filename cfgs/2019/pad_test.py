import os
import csv
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def init(mdlParams_):
    mdlParams = {}
    
    # 1. 경로 설정
    mdlParams['saveDir'] = mdlParams_['pathBase'] + '/results/'
    mdlParams['dataDir'] = mdlParams_['pathBase'] + '/data'
    img_folder = os.path.join(mdlParams['dataDir'], 'imgs')
    csv_path = os.path.join(mdlParams['dataDir'], 'metadata.csv')
    
    ### 기본 모델 및 클래스 설정 ###
    mdlParams['model_type'] = 'efficientnet-b0'
    mdlParams['numClasses'] = 4
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 5
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['file_ending'] = '.png'

    ### [중요] utils.py의 ISICDataset 클래스가 요구하는 필수 키값들 ###
    mdlParams['preload'] = False           # 이미지 미리 로드 여부
    mdlParams['orderedCrop'] = False       # 평가 시 정해진 위치 크롭 여부
    mdlParams['multiCropEval'] = 0         # 평가 시 멀티 크롭 개수 (0은 단일 이미지)
    mdlParams['same_sized_crops'] = True   # 학습 시 RandomCrop 사용 여부
    mdlParams['balance_classes'] = 0       # 클래스 균형 옵션 (0: 기본)
    mdlParams['subtract_set_mean'] = False # 평균값 차감 여부
    mdlParams['setMean'] = np.array([0.485, 0.456, 0.406]) # ImageNet 평균
    mdlParams['setStd'] = np.array([0.229, 0.224, 0.225])  # ImageNet 표준편차
    mdlParams['trainSetState'] = 'train'   # 현재 상태 (train/eval)
    mdlParams['numGPUs'] = [0]
    
    
    # 추가로 utils.py에서 get() 없이 참조하는 키들
    mdlParams['only_downsmaple'] = False   # Resize만 할지 여부
    mdlParams['classification'] = True     # 분류 모드
    mdlParams['print_trainerr'] = False
    mdlParams['peak_at_testerr'] = False
    mdlParams['voting_scheme'] = 'average' # 멀티크롭 시 투표 방식

    ### 학습 하이퍼파라미터 ###
    mdlParams['batchSize'] = 16 
    mdlParams['learning_rate'] = 0.000015
    mdlParams['lowerLRAfter'] = 20
    mdlParams['lowerLRat'] = 40
    mdlParams['LRstep'] = 5
    mdlParams['training_steps'] = 1
    mdlParams['display_step'] = 10
    
    # 데이터 증강 설정 (utils.py의 증강 로직 활성화)
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8, 1.2)
    mdlParams['cutout'] = 0                # 필요 시 숫자로 조절 (예: 16)
    mdlParams['autoaugment'] = False

    ### Data Loading & Filtering ###
    diag_map = {'NEV': 0, 'BCC': 1, 'SCC': 2, 'MEL': 3}
    
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    patient_ids = []     # 분할용 그룹 정보
    class_indices = []   # 분할용 계층 정보
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            diag_name = row['diagnostic']
            
            # 4개 클래스만 필터링
            if diag_name in diag_map:
                img_id = row['img_id']
                img_path = os.path.join(img_folder, img_id)
                
                if os.path.exists(img_path):
                    mdlParams['im_paths'].append(img_path)
                    
                    # 라벨 생성
                    label = np.zeros(mdlParams['numClasses'])
                    label[diag_map[diag_name]] = 1
                    mdlParams['labels_list'].append(label)
                    
                    # 분할용 메타 정보 수집
                    patient_ids.append(row['patient_id'])
                    class_indices.append(diag_map[diag_name])

    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    patient_ids = np.array(patient_ids)
    class_indices = np.array(class_indices)
    img_indices = np.arange(len(mdlParams['im_paths']))

    print(f"--- 데이터 필터링 완료 ---")
    print(f"학습 대상 이미지 총 개수: {len(mdlParams['im_paths'])}")

    ### [핵심] Patient-wise Stratified 5-Fold 분할 ###
    # n_splits=5로 설정하면 자동으로 80%(Train) : 20%(Val)로 나뉩니다.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    mdlParams['trainIndCV'] = []
    mdlParams['valIndCV'] = []

    for train_idx, val_idx in sgkf.split(img_indices, class_indices, groups=patient_ids):
        mdlParams['trainIndCV'].append(train_idx)
        mdlParams['valIndCV'].append(val_idx)

    print("--- 5-Fold (8:2) 분할 완료 ---")
    for i in range(len(mdlParams['trainIndCV'])):
        t_num = len(mdlParams['trainIndCV'][i])
        v_num = len(mdlParams['valIndCV'][i])
        print(f"Fold {i}: Train {t_num}장 | Val {v_num}장")
    
    mdlParams['orderedCrop'] = False
    mdlParams['multiCropEval'] = 0
    
    return mdlParams