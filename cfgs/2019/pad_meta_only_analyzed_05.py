import os
import csv
import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def init(mdlParams_):
    mdlParams = {}
    
    # 경로 설정
    mdlParams['saveDir'] = mdlParams_['pathBase'] + '/results/'
    mdlParams['dataDir'] = mdlParams_['pathBase'] + '/data'
    img_folder = os.path.join(mdlParams['dataDir'], 'folds/fold0/imgs')
    train_csv = os.path.join(mdlParams['dataDir'], 'folds/fold0/train.csv')
    val_csv = os.path.join(mdlParams['dataDir'], 'folds/fold0/val.csv')
    
    # 모델 기본 설정
    mdlParams['model_type'] = 'meta-only'
    mdlParams['numClasses'] = 4
    mdlParams['numCV'] = 5
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['file_ending'] = '.png'
    mdlParams['orderedCrop'] = False
    mdlParams['multiCropEval'] = 0
    mdlParams['same_sized_crops'] = True
    mdlParams['subtract_set_mean'] = False
    mdlParams['crop_size'] = [224, 224]
    mdlParams['drop_augment'] = 0.1  
    mdlParams['multiCropEval'] = 0
    mdlParams['same_sized_crops'] = True
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.485, 0.456, 0.406])
    mdlParams['setStd'] = np.array([0.229, 0.224, 0.225])
    mdlParams['preload'] = False
    mdlParams['balance_classes'] = 0
    mdlParams['numRandValSeq'] = 0
    mdlParams['eval_flipping'] = 0

    # 메타데이터 추가 설정
    mdlParams['meta_features'] = ['age', 'gender', 'region',
                                  'itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation',
                                  'smoke', 'skin_cancer_history'
                                  ]
    mdlParams['meta_feature_sizes'] = [1, 2, 14,
                                        2, 2, 2, 2, 2, 2,
                                        2, 2]
    mdlParams['encode_nan'] = False
    mdlParams['freeze_cnn'] = True
    mdlParams['learning_rate_meta'] = 0.00001
    mdlParams['dropout_meta'] = 0.4
    mdlParams['scale_features'] = True
    mdlParams['fc_layers_before'] = [256, 256]

    mdlParams['preload'] = False
    mdlParams['subtract_set_mean'] = False
    mdlParams['balance_classes'] = 0
    mdlParams['multiCropEval'] = 0
    mdlParams['numGPUs'] = [0]

    mdlParams['only_downsmaple'] = False
    mdlParams['classification'] = True
    mdlParams['print_trainerr'] = False
    mdlParams['peak_at_testerr'] = False
    mdlParams['voting_scheme'] = 'average'

    mdlParams['batchSize'] = 32        
    mdlParams['learning_rate'] = 0.0001
    mdlParams['lowerLRAfter'] = 20
    mdlParams['lowerLRat'] = 40
    mdlParams['LRstep'] = 5
    mdlParams['epoch'] = 100

    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8, 1.2)
    mdlParams['cutout'] = 0
    mdlParams['autoaugment'] = False

    # --- 데이터 파싱 로직 ---
    diag_map = {'NEV': 0, 'BCC': 1, 'SCC': 2, 'MEL': 3}
    gender_map = {'MALE': 0, 'FEMALE': 1}
    region_map = {
        'FACE': 0, 'BACK': 1, 'CHEST': 2, 'NOSE': 3, 'FOREARM': 4, 
        'ARM': 5, 'NECK': 6, 'EAR': 7, 'HAND': 8, 'THIGH': 9,
        'ABDOMEN': 10, 'LIP': 11, 'FOOT': 12, 'SCALP': 13
    }
    
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    mdlParams['meta_dict'] = {} 
    
    current_idx = 0
    
    for csv_file, is_train in [(train_csv, True), (val_csv, False)]:
        indices_for_this_file = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                diag_name = row['diagnostic']
                if diag_name in diag_map:
                    img_id = row['img_id']
                    img_path = os.path.join(img_folder, img_id)
                    
                    if os.path.exists(img_path):
                        mdlParams['im_paths'].append(img_path) 
                        label = np.zeros(mdlParams['numClasses'])
                        label[diag_map[diag_name]] = 1
                        mdlParams['labels_list'].append(label)
                        
                        indices_for_this_file.append(current_idx)
                        current_idx += 1

                        # --- 메타데이터 벡터 생성 로직 ---
                        feature_vector = []
                        
                        # 1. 나이 (Age): 단순 수치
                        age = float(row.get('age', 0))
                        feature_vector.append(np.array([age]))

                        # 2. 성별 (Gender): One-hot (2차원)
                        gender_vec = np.zeros(2)
                        g_idx = gender_map.get(row.get('gender', '').upper(), 0) # 기본값 0
                        gender_vec[g_idx] = 1.0
                        feature_vector.append(gender_vec)

                        # 3. 부위 (Region): One-hot (14차원)
                        region_vec = np.zeros(14)
                        r_idx = region_map.get(row.get('region', '').upper(), 0) # 기본값 0
                        region_vec[r_idx] = 1.0
                        feature_vector.append(region_vec)

                        symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation',
                                    'smoke', 'skin_cancer_history'
                                    ]
                        
                        for s in symptoms:
                            raw_val = row.get(s, 'unk').lower()
                            bool_vec = np.zeros(2) # 기본 [0, 0] (unk일 때)
                            
                            if raw_val == 'true':
                                bool_vec[1] = 1.0  # [0, 1]
                            elif raw_val == 'false':
                                bool_vec[0] = 1.0  # [1, 0]
                            # unk이거나 그 외의 경우 [0, 0] 유지
                            
                            feature_vector.append(bool_vec)

                        # 최종 벡터 합치기 총 35차원
                        meta_vector = np.concatenate(feature_vector).astype(np.float32)
                        mdlParams['meta_dict'][img_path] = meta_vector
        if is_train:
            mdlParams['trainIndCV'] = [np.array(indices_for_this_file)]
        else:
            mdlParams['valIndCV'] = [np.array(indices_for_this_file)]

    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    mdlParams['meta_array'] = np.array([mdlParams['meta_dict'][p] for p in mdlParams['im_paths']])

    # 3. 클래스 가중치 계산 (증강된 Train 기준)
    train_indices = mdlParams['trainIndCV'][0]
    # 전체 labels_array에서 train에 해당하는 것만 추출해서 클래스 분포 확인
    train_labels = mdlParams['labels_array'][train_indices]
    class_counts = np.sum(train_labels, axis=0)
    
    weights = len(train_indices) / (mdlParams['numClasses'] * class_counts)
    mdlParams['class_weights'] = torch.FloatTensor(weights)

    print(f"--- 로드 완료 (Fold 0) ---")
    print(f"Train (증강포함): {len(mdlParams['trainIndCV'][0])}장")
    print(f"Val (원본): {len(mdlParams['valIndCV'][0])}장")
    print(f"가중치: {weights}")

    return mdlParams