import importlib
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from torchvision import transforms

# 기존 프로젝트 모듈 임포트
import utils
import models

def inference(img_name, meta_dict):
    '''
    img_name: 이미지 파일 이름 (예: 'test.jpg')
    meta_dict: {'age': 50, 'gender': 'MALE', 'region': 'FACE'} 형태
    '''
    mdlParams = {}

    # 설정 로드
    pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    # GPU 환경 설정
    if 'mps' in sys.argv[3]:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    elif 'gpu' in sys.argv[3] or 'cuda' in sys.argv[3]:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # 모델 초기화 및 가중치 로드
    model = models.getModel(mdlParams)()
    
    if 'efficient' in mdlParams['model_type']:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])

    if mdlParams.get('meta_features', None) is not None:
        model = models.modify_meta(mdlParams, model)

    # 가중치 불러오기
    chkPath = 'results/2019.pad_meta/meta_baseline_normalized/best_fold0.pt' # 주석의 경로
    if os.path.isfile(chkPath):
        state = torch.load(chkPath, map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        print(f"Loaded weights from {chkPath}")
    else:
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없어: {chkPath}")

    model.to(device)
    model.eval()

    # 입력 데이터 전처리
    input_img = Image.open(img_name).convert('RGB')
    
    input_img = Image.open(img_name).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(mdlParams['input_size'][:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(input_img).unsqueeze(0).to(device)

    # 메타데이터 전처리
    if mdlParams.get('meta_features', None) is not None:
        gender_map = {'MALE': 0, 'FEMALE': 1}
        region_map = {
            'FACE': 0, 'BACK': 1, 'CHEST': 2, 'NOSE': 3, 'FOREARM': 4, 
            'ARM': 5, 'NECK': 6, 'EAR': 7, 'HAND': 8, 'THIGH': 9,
            'ABDOMEN': 10, 'LIP': 11, 'FOOT': 12, 'SCALP': 13
        }

        feature_vector = []
        
        # 나이 
        age = float(meta_dict.get('age', 0))
        age = age / 100.0
        feature_vector.append(np.array([age]))

        # 성별
        gender_vec = np.zeros(2)
        g_idx = gender_map.get(str(meta_dict.get('gender', '')).upper(), 0)
        gender_vec[g_idx] = 1.0
        feature_vector.append(gender_vec)

        # 부위
        region_vec = np.zeros(14)
        r_idx = region_map.get(str(meta_dict.get('region', '')).upper(), 0)
        region_vec[r_idx] = 1.0
        feature_vector.append(region_vec)

        # 최종 17차원 벡터 생성
        meta_array = np.concatenate(feature_vector).astype(np.float32)
        
        # 스케일러 적용
        if mdlParams.get('scale_features', False) and 'feature_scaler_meta' in mdlParams:
            meta_array = mdlParams['feature_scaler_meta'].transform(meta_array.reshape(1, -1))
        
        meta_tensor = torch.from_numpy(meta_array).reshape(1, -1).to(device)
        inputs = [img_tensor, meta_tensor]
    else:
        inputs = img_tensor

    # 추론
    with torch.no_grad():
        outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        conf, pred = torch.max(probs, 1)

    class_names = ['NEV', 'BCC', 'SCC', 'MEL']
    predicted_class = class_names[pred.item()]
    probability = conf.item()

    return predicted_class, probability

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python inference.py [pc_cfg] [model_cfg] [device]")
        sys.exit()

    # 테스트용 데이터 mel
    test_meta = {
        'age': 21,
        'gender': 'MALE',
        'region': 'BACK'
    }

    result_class, result_prob = inference('test_nev.png', test_meta)
    print(f"결과: {result_class} (확률: {result_prob:.4f})")