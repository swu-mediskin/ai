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
    mdlParams['model_type'] = 'efficientnet-b0'
    mdlParams['numClasses'] = 4
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 5
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['file_ending'] = '.png'

    # 메타데이터 추가 설정
    mdlParams['meta_features'] = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']
    mdlParams['meta_feature_sizes'] = [2, 2, 2, 2, 2, 2] 
    mdlParams['encode_nan'] = False
    mdlParams['freeze_cnn'] = True
    mdlParams['learning_rate_meta'] = 0.00001
    mdlParams['dropout_meta'] = 0.4
    mdlParams['scale_features'] = True
    mdlParams['drop_augment'] = 0.1
    mdlParams['model_load_path'] = mdlParams_['pathBase'] + '/results/image_baseline'
    mdlParams['fc_layers_before'] = [256, 256] 
    mdlParams['fc_layers_after'] = [1024]

    mdlParams['preload'] = False           
    mdlParams['orderedCrop'] = False       
    mdlParams['multiCropEval'] = 0         
    mdlParams['same_sized_crops'] = True   
    mdlParams['balance_classes'] = 0       
    mdlParams['subtract_set_mean'] = False 
    mdlParams['setMean'] = np.array([0.485, 0.456, 0.406]) 
    mdlParams['setStd'] = np.array([0.229, 0.224, 0.225])    
    mdlParams['numGPUs'] = [0]
    
    mdlParams['only_downsmaple'] = False   
    mdlParams['classification'] = True     
    mdlParams['print_trainerr'] = False
    mdlParams['peak_at_testerr'] = False
    mdlParams['voting_scheme'] = 'average' 

    # 학습 하이퍼파라미터 
    mdlParams['batchSize'] = 32 
    mdlParams['learning_rate'] = 0.0001
    mdlParams['lowerLRAfter'] = 20
    mdlParams['lowerLRat'] = 40
    mdlParams['LRstep'] = 5
    mdlParams['epoch'] = 100
    
    # 데이터 증강 설정 
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8, 1.2)
    mdlParams['cutout'] = 0                
    mdlParams['autoaugment'] = False

    diag_map = {'NEV': 0, 'BCC': 1, 'SCC': 2, 'MEL': 3}
    
    
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

                    # --- 메타데이터 벡터 생성 로직 ---
                    feature_vector = []
                    
                    symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation']
                    
                    for s in symptoms:
                        raw_val = row.get(s, 'unk').lower()
                        bool_vec = np.zeros(2) # 기본 [0, 0] (unk일 때)
                        
                        if raw_val == 'true':
                            bool_vec[1] = 1.0  # [0, 1]
                        elif raw_val == 'false':
                            bool_vec[0] = 1.0  # [1, 0]
                        # unk이거나 그 외의 경우 [0, 0] 유지
                        
                        feature_vector.append(bool_vec)

                    # 최종 벡터 합치기 (2 * 6 = 12차원)
                    meta_vector = np.concatenate(feature_vector).astype(np.float32)
                    mdlParams['meta_dict'][img_path] = meta_vector

    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    mdlParams['meta_array'] = np.array([mdlParams['meta_dict'][p] for p in mdlParams['im_paths']])

    patient_ids = np.array(patient_ids)
    class_indices = np.array(class_indices)
    img_indices = np.arange(len(mdlParams['im_paths']))

    print(f"--- 데이터 필터링 완료 ---")
    print(f"학습 대상 이미지 총 개수: {len(mdlParams['im_paths'])}")
    print(f"메타데이터 벡터 차원: {mdlParams['meta_array'].shape[1]}")

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

    unique, counts = np.unique(class_indices, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print("\n--- 클래스별 데이터 분포 ---")
    for cls, count in class_counts.items():
        inv_diag_map = {v: k for k, v in diag_map.items()}
        print(f"{inv_diag_map[cls]}: {count}장")

    total_samples = len(class_indices)
    num_classes = mdlParams['numClasses']
    
    weights = [total_samples / (num_classes * class_counts[i]) if i in class_counts else 0 for i in range(num_classes)]
    
    mdlParams['class_weights'] = torch.FloatTensor(weights)
    
    print("\n--- 계산된 클래스 가중치 ---")
    print(weights)

    return mdlParams