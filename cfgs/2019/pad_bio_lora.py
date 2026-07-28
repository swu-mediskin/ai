import os
import re
import torch
import numpy as np
import pandas as pd

import models
from meta2text import row2text


def init(mdlParams_):
    mdlParams = {}

    # 경로 설정
    mdlParams['saveDir'] = mdlParams_['pathBase'] + '/results/'
    mdlParams['dataDir'] = mdlParams_['pathBase'] + '/data'
    img_folder = os.path.join(mdlParams['dataDir'], 'folds/fold0/imgs')
    train_csv = os.path.join(mdlParams['dataDir'], 'folds/fold0/train.csv')
    val_csv = os.path.join(mdlParams['dataDir'], 'folds/fold0/val.csv')

    # 이름 설정
    mdlParams['mdlName'] = 'bio_lora_vit2'  # 템플릿 사용!!

    # 모델 기본 설정
    mdlParams['model_type'] = 'biomedclip'
    mdlParams['numClasses'] = 4
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 1
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['file_ending'] = '.png'

    mdlParams['freeze_biomedclip'] = True
    mdlParams['unfreeze_visual_blocks'] = 0
    mdlParams['unfreeze_text_blocks'] = 0
    mdlParams['use_lora_visual'] = True
    mdlParams['lora_visual_blocks'] = 2
    mdlParams['use_lora_text'] = False
    mdlParams['lora_text_blocks'] = 0
    mdlParams['lora_r'] = 8
    mdlParams['lora_alpha'] = 16  # 2*r, 표준 관례
    mdlParams['lora_dropout'] = 0.05
    mdlParams['learning_rate_meta'] = 0.001
    mdlParams['dropout_meta'] = 0.4
    mdlParams['fc_layers_after'] = [1024]

    # SCC를 BCC로(혹은 그 반대로) 착각하는 오분류에 weighted CE보다 더 강한 페널티를 추가.
    mdlParams['use_cost_sensitive_ce'] = True
    mdlParams['cost_sensitive_pairs'] = [('SCC', 'BCC', 3.0)]
    mdlParams['cost_sensitive_lambda'] = 1.0

    mdlParams['meta_features'] = ['biomedclip_text']
    mdlParams['meta_feature_sizes'] = [models.BIOMEDCLIP_CONTEXT_LENGTH]
    mdlParams['scale_features'] = False
    mdlParams['drop_augment'] = 0

    mdlParams['preload'] = False
    mdlParams['orderedCrop'] = False
    mdlParams['multiCropEval'] = 0
    mdlParams['same_sized_crops'] = True
    mdlParams['balance_classes'] = 0
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.48145466, 0.4578275, 0.40821073])
    mdlParams['setStd'] = np.array([0.26862954, 0.26130258, 0.27577711])
    mdlParams['numGPUs'] = [0]

    mdlParams['only_downsmaple'] = False
    mdlParams['classification'] = True
    mdlParams['print_trainerr'] = False
    mdlParams['peak_at_testerr'] = False
    mdlParams['voting_scheme'] = 'average'

    # 학습 하이퍼파라미터
    mdlParams['batchSize'] = 32
    mdlParams['learning_rate_lora'] = 0.0003
    mdlParams['learning_rate'] = 0.000001
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

    tokenizer = models.get_biomedclip_tokenizer()
    current_idx = 0

    for csv_file, is_train in [(train_csv, True), (val_csv, False)]:
        indices_for_this_file = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            diag_name = row['diagnostic']
            if diag_name in diag_map:
                img_id = row['img_id']
                img_path = os.path.join(img_folder, img_id)

                mdlParams['im_paths'].append(img_path)
                label = np.zeros(mdlParams['numClasses'])
                label[diag_map[diag_name]] = 1
                mdlParams['labels_list'].append(label)

                indices_for_this_file.append(current_idx)
                current_idx += 1

                text = row2text(row)
                tokens = tokenizer([text], context_length=models.BIOMEDCLIP_CONTEXT_LENGTH)
                mdlParams['meta_dict'][img_path] = tokens[0].numpy().astype(np.float32)
        if is_train:
            mdlParams['trainIndCV'] = [np.array(indices_for_this_file)]
        else:
            mdlParams['valIndCV'] = [np.array(indices_for_this_file)]

    mdlParams['labels_array'] = np.array(mdlParams['labels_list'])
    mdlParams['meta_array'] = np.array([mdlParams['meta_dict'][p] for p in mdlParams['im_paths']])

    # 클래스 가중치 계산 (train 기준, aug_400.py가 만든 증강 이미지는 제외하고 원본 데이터 분포로 계산)
    train_indices = np.array(mdlParams['trainIndCV'][0]).astype(int)
    is_original = np.array([
        re.search(r'_aug\d+\.', os.path.basename(mdlParams['im_paths'][idx])) is None
        for idx in train_indices
    ])
    original_train_indices = train_indices[is_original]
    train_labels = mdlParams['labels_array'][original_train_indices]
    class_counts = np.sum(train_labels, axis=0)

    weights = len(original_train_indices) / (mdlParams['numClasses'] * class_counts)
    mdlParams['class_weights'] = torch.FloatTensor(np.array(weights).flatten())

    print(f"--- 로드 완료 (Fold 0, BiomedCLIP, 템플릿 메타데이터 문장, LoRA last 2 visual blocks only) ---")
    print(f"Train: {len(mdlParams['trainIndCV'][0])}장")
    print(f"Val: {len(mdlParams['valIndCV'][0])}장")
    print(f"가중치: {weights}")

    return mdlParams
