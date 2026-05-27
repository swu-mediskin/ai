import os
import csv
import torch
import numpy as np

def init(mdlParams_):
    mdlParams = {}
    
    # 경로 설정
    mdlParams['saveDir'] = mdlParams_['pathBase'] + '/results/'
    mdlParams['dataDir'] = mdlParams_['pathBase'] + '/data'
    img_folder = os.path.join(mdlParams['dataDir'], 'mel/fold0/imgs')
    train_csv = os.path.join(mdlParams['dataDir'], 'mel/fold0/train.csv')
    val_csv = os.path.join(mdlParams['dataDir'], 'mel/fold0/val.csv')
    
    # 모델 기본 설정
    mdlParams['model_type'] = 'efficientnet-b0'
    mdlParams['numClasses'] = 4
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 1  
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['file_ending'] = '.png'

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

    diag_map = {'NEV': 0, 'BCC': 1, 'SCC': 2, 'MEL': 3}
    
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    class_indices = []
    train_indices = []
    val_indices = []

    # train.csv 읽기
    with open(train_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diag_name = row['diagnostic']
            if diag_name in diag_map:
                img_path = os.path.join(img_folder, row['img_id'])
                if os.path.exists(img_path):
                    train_indices.append(len(mdlParams['im_paths']))
                    mdlParams['im_paths'].append(img_path)
                    label = np.zeros(mdlParams['numClasses'])
                    label[diag_map[diag_name]] = 1
                    mdlParams['labels_list'].append(label)
                    class_indices.append(diag_map[diag_name])

    # val.csv 읽기 (원본만)
    with open(val_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diag_name = row['diagnostic']
            if diag_name in diag_map:
                img_path = os.path.join(img_folder, row['img_id'])
                if os.path.exists(img_path):
                    val_indices.append(len(mdlParams['im_paths']))
                    mdlParams['im_paths'].append(img_path)
                    label = np.zeros(mdlParams['numClasses'])
                    label[diag_map[diag_name]] = 1
                    mdlParams['labels_list'].append(label)
                    class_indices.append(diag_map[diag_name])

    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    class_indices = np.array(class_indices)

    # fold0만 사용
    mdlParams['trainIndCV'] = [np.array(train_indices)]
    mdlParams['valIndCV'] = [np.array(val_indices)]

    print(f"--- 데이터 로드 완료 ---")
    print(f"Train: {len(train_indices)}장 | Val: {len(val_indices)}장")

    # class_weights는 train 기준으로 계산
    train_class_indices = class_indices[train_indices]
    unique, counts = np.unique(train_class_indices, return_counts=True)
    class_counts = dict(zip(unique, counts))

    inv_diag_map = {v: k for k, v in diag_map.items()}
    print("\n--- Train 클래스별 데이터 분포 ---")
    for cls, count in class_counts.items():
        print(f"{inv_diag_map[cls]}: {count}장")

    total_samples = len(train_indices)
    weights = [total_samples / (mdlParams['numClasses'] * class_counts[i]) if i in class_counts else 0 for i in range(mdlParams['numClasses'])]
    mdlParams['class_weights'] = torch.FloatTensor(weights)

    print("\n--- 계산된 클래스 가중치 ---")
    print(weights)

    return mdlParams