import importlib
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import sklearn.preprocessing
from torchvision import transforms
import models

def build_meta_vector(meta_dict, mdlParams, scaler=None):
    """메타데이터 dict → 전처리된 numpy array 반환"""
    gender_map = {'MALE': 0, 'FEMALE': 1}
    region_map = {
        'FACE': 0, 'BACK': 1, 'CHEST': 2, 'NOSE': 3, 'FOREARM': 4,
        'ARM': 5, 'NECK': 6, 'EAR': 7, 'HAND': 8, 'THIGH': 9,
        'ABDOMEN': 10, 'LIP': 11, 'FOOT': 12, 'SCALP': 13
    }

    feature_vector = []
    feature_vector.append(np.array([float(meta_dict.get('age', 0))]))

    gender_vec = np.zeros(2)
    gender_vec[gender_map.get(str(meta_dict.get('gender', '')).upper(), 0)] = 1.0
    feature_vector.append(gender_vec)

    region_vec = np.zeros(14)
    region_vec[region_map.get(str(meta_dict.get('region', '')).upper(), 0)] = 1.0
    feature_vector.append(region_vec)

    for key in ['fitspatrick', 'diameter_1', 'diameter_2']:
        val = str(meta_dict.get(key, '')).strip()
        feature_vector.append(np.array([float(val) if val else 0.0]))

    symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation',
                'smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history']
    for s in symptoms:
        raw_val = str(meta_dict.get(s, 'unk')).lower()
        bool_vec = np.zeros(2)
        if raw_val == 'true': bool_vec[1] = 1.0
        elif raw_val == 'false': bool_vec[0] = 1.0
        feature_vector.append(bool_vec)

    meta_array = np.concatenate(feature_vector).astype(np.float32).reshape(1, -1)

    print(f"\n📋 META 벡터 변환")
    print(f"  입력: {meta_dict}")
    print(f"  raw 벡터    (dim={meta_array.shape[1]}): {np.round(meta_array[0], 3).tolist()}")

    if scaler is not None and mdlParams.get('scale_features'):
        meta_array = scaler.transform(meta_array)
        print(f"  scaled 벡터 (dim={meta_array.shape[1]}): {np.round(meta_array[0], 3).tolist()}")

    return meta_array


def inference(img_name, meta_dict):
    mdlParams = {}

    cfg_name = sys.argv[1].replace('.py', '')
    pc_cfg = importlib.import_module('pc_cfgs.' + cfg_name)
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.' + sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    if 'mps' in sys.argv[3]:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    elif 'gpu' in sys.argv[3] or 'cuda' in sys.argv[3]:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = models.getModel(mdlParams)()
    if 'efficient' in mdlParams['model_type']:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
    if mdlParams.get('meta_features', None) is not None:
        model = models.modify_meta(mdlParams, model)

    chkPath = 'results/2019.pad_multi_all_real_aug/multi_all_real_aug/best_fold0.pt'
    if os.path.isfile(chkPath):
        state = torch.load(chkPath, map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        print(f"✅ 가중치 로드 완료: {chkPath}")
    else:
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없어: {chkPath}")

    model.to(device)
    model.eval()

    input_img = Image.open(img_name).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(mdlParams['input_size'][:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mdlParams['setMean'], std=mdlParams['setStd'])
    ])
    img_tensor = preprocess(input_img).unsqueeze(0).to(device)

    if mdlParams.get('meta_features', None) is not None:
        scaler = sklearn.preprocessing.StandardScaler().fit(
            mdlParams['meta_array'][mdlParams['trainIndCV'][0], :]
        )
        meta_array = build_meta_vector(meta_dict, mdlParams, scaler)
        meta_tensor = torch.from_numpy(meta_array).float().to(device)
        inputs = [img_tensor, meta_tensor]
    else:
        inputs = img_tensor

    with torch.no_grad():
        outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        print(f"DEBUG - 클래스별 확률: {probs.cpu().numpy()}")
        conf, pred = torch.max(probs, 1)

    class_names = ['NEV', 'BCC', 'SCC', 'MEL']
    return class_names[pred.item()], conf.item()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python inference.py [pc_cfg] [model_cfg] [device]")
        sys.exit()

    test_meta = {
        'age': 65.0, 'gender': 'MALE', 'region': 'BACK',
        'fitspatrick': 3.0, 'diameter_1': 15.0, 'diameter_2': 12.0,
        'itch': 'true', 'grew': 'true', 'hurt': 'false', 'changed': 'true',
        'bleed': 'true', 'elevation': 'true', 'smoke': 'true', 'drink': 'true',
        'pesticide': 'false', 'skin_cancer_history': 'false', 'cancer_history': 'true'
    }

    result_class, result_prob = inference('test_bcc.png', test_meta)
    print(f"\n📢 최종 결과: {result_class} (확률: {result_prob:.4f})")