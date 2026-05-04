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
    img_folder = os.path.join(mdlParams['dataDir'], 'imgs')
    csv_path = os.path.join(mdlParams['dataDir'], 'metadata.csv')
    
    # 모델 기본 설정
    mdlParams['model_type'] = 'meta-only'
    mdlParams['numClasses'] = 4
    mdlParams['numCV'] = 5
    mdlParams['input_size'] = [224, 224, 3] 
    mdlParams['input_size'] = [224, 224, 3] 
    mdlParams['orderedCrop'] = False
    mdlParams['multiCropEval'] = 0
    mdlParams['same_sized_crops'] = True
    mdlParams['subtract_set_mean'] = False
    mdlParams['crop_size'] = [224, 224]
    mdlParams['drop_augment'] = 0.0
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
                                  'fitspatrick', 'diameter_1', 'diameter_2',
                                  'itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation',
                                  'smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history'
                                  ]
    
    # 메타 전용 학습 설정
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
    
    # 학습 하이퍼파라미터 
    mdlParams['batchSize'] = 64 
    mdlParams['epoch'] = 500     
    mdlParams['lowerLRAfter'] = 20 
    mdlParams['lowerLRat'] = 40 
    mdlParams['LRstep'] = 5

    # --- 데이터 파싱 로직
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
    patient_ids = []     
    class_indices = []   
    
    with open(csv_path, newline='') as csvfile:
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
                    patient_ids.append(row['patient_id'])
                    class_indices.append(diag_map[diag_name])

                    feature_vector = []
                    # 나이
                    age = float(row.get('age', 0))
                    feature_vector.append(np.array([age]))
                    # 성별
                    gender_vec = np.zeros(2)
                    g_idx = gender_map.get(row.get('gender', '').upper(), 0)
                    gender_vec[g_idx] = 1.0
                    feature_vector.append(gender_vec)
                    # 부위
                    region_vec = np.zeros(14)
                    r_idx = region_map.get(row.get('region', '').upper(), 0)
                    region_vec[r_idx] = 1.0
                    feature_vector.append(region_vec)
                    # 기타 수치형
                    feature_vector.append(np.array([float(row.get('fitspatrick', 0) or 0)]))
                    feature_vector.append(np.array([float(row.get('diameter_1', 0) or 0)]))
                    feature_vector.append(np.array([float(row.get('diameter_2', 0) or 0)]))

                    # 증상 (Binary)
                    symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation',
                                'smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history']
                    for s in symptoms:
                        raw_val = row.get(s, 'unk').lower()
                        bool_vec = np.zeros(2)
                        if raw_val == 'true': bool_vec[1] = 1.0
                        elif raw_val == 'false': bool_vec[0] = 1.0
                        feature_vector.append(bool_vec)

                    meta_vector = np.concatenate(feature_vector).astype(np.float32)
                    mdlParams['meta_dict'][img_path] = meta_vector

    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    mdlParams['meta_array'] = np.array([mdlParams['meta_dict'][p] for p in mdlParams['im_paths']])

    # CV 분할 및 가중치 계산 
    patient_ids = np.array(patient_ids)
    class_indices = np.array(class_indices)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    mdlParams['trainIndCV'] = []
    mdlParams['valIndCV'] = []
    for train_idx, val_idx in sgkf.split(np.arange(len(mdlParams['im_paths'])), class_indices, groups=patient_ids):
        mdlParams['trainIndCV'].append(train_idx)
        mdlParams['valIndCV'].append(val_idx)

    unique, counts = np.unique(class_indices, return_counts=True)
    class_counts = dict(zip(unique, counts))
    weights = [len(class_indices) / (mdlParams['numClasses'] * class_counts[i]) for i in range(mdlParams['numClasses'])]
    mdlParams['class_weights'] = torch.FloatTensor(weights)

    return mdlParams