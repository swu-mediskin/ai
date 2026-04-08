import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
# from torchsummary import summary # 필요 시 설치 후 주석 해제
import numpy as np
from scipy import io
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import importlib
import time
import sklearn.preprocessing
import utils
import psutil
import models


if __name__ == '__main__':

        
    # 1. 설정 파일 로드
    mdlParams = {}
    # 예: python train.py example 2019.test_effb0_ss mps 0
    pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    mdlParams['trainSetState'] = 'train'
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]

    # 2. 디바이스 설정 (MPS 지원 추가)
    if 'mps' in sys.argv[3]:
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif 'gpu' in sys.argv[3] or 'cuda' in sys.argv[3]:
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 3. 데이터 및 결과 저장소 초기화
    if len(sys.argv) > 4:
        cv_set = [int(s) for s in re.findall(r'\d+', sys.argv[4])]
    else:
        cv_set = range(mdlParams['numCV'])

    prevFile = Path(mdlParams['saveDirBase'] + '/CV.pkl')
    allData = pickle.load(open(prevFile, 'rb')) if prevFile.exists() else {
        k: {} for k in ['f1Best', 'sensBest', 'specBest', 'accBest', 'waccBest', 'aucBest', 'convergeTime', 'bestPred', 'targets']
    }

    # 4. Fold별 학습 루프
    for cv in cv_set:
        print(f"\n--- Starting Fold {cv} ---")
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
        if not os.path.isdir(mdlParams['saveDirBase']): os.makedirs(mdlParams['saveDirBase'], exist_ok=True)
        if not os.path.isdir(mdlParams['saveDir']): os.makedirs(mdlParams['saveDir'], exist_ok=True)

        # 인덱스 설정 (cfg에서 이미 생성됨)
        mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
        eval_set = 'valInd'

        # 모델 초기화
        importlib.reload(models)
        modelVars = {'device': device}
        modelVars['model'] = models.getModel(mdlParams)()
        modelVars['softmax'] = nn.Softmax(dim=1)

        # 모델 마지막 레이어 수정 (EfficientNet-B0 기준)
        if 'efficient' in mdlParams['model_type']:
            num_ftrs = modelVars['model']._fc.in_features
            modelVars['model']._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
        
        modelVars['model'] = modelVars['model'].to(device)

        # 5. 데이터로더 설정
        num_workers = psutil.cpu_count(logical=False)
        dataset_train = utils.ISICDataset(mdlParams, 'trainInd')
        dataset_val = utils.ISICDataset(mdlParams, 'valInd')
        
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=True)

        # 6. 손실 함수 및 옵티마이저
        # 4개 클래스 단순 분류이므로 CrossEntropy 사용
        modelVars['criterion'] = nn.CrossEntropyLoss()
        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])
        modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/mdlParams['LRstep'])

        # 7. 학습 루프
        mdlParams['valBest'] = 1000
        start_time = time.time()
        
        for step in range(1, mdlParams['training_steps'] + 1):
            modelVars['model'].train()
            for j, (inputs, labels, indices) in enumerate(modelVars['dataloader_trainInd']):
                inputs, labels = inputs.to(device), labels.to(device)
                
                modelVars['optimizer'].zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = modelVars['model'](inputs)
                    loss = modelVars['criterion'](outputs, labels)
                    loss.backward()
                    modelVars['optimizer'].step()

            # 스케줄러 업데이트
            if step >= mdlParams['lowerLRat'] - mdlParams['lowerLRAfter']:
                modelVars['scheduler'].step()

            # 평가 및 저장
            if step % mdlParams['display_step'] == 0 or step == 1:
                modelVars['model'].eval()
                # utils.getErrClassification_mgpu 함수 내에서 .cuda()를 사용하는지 확인 필요 (mps로 수정 권장)
                res = utils.getErrClassification_mgpu(mdlParams, eval_set, modelVars)
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = res
                
                eval_metric = -np.mean(waccuracy)
                if eval_metric < mdlParams['valBest']:
                    mdlParams['valBest'] = eval_metric
                    # 최적 모델 저장 로직 (생략 가능하나 유지는 권장)
                    state = {'epoch': step, 'state_dict': modelVars['model'].state_dict(), 'optimizer': modelVars['optimizer'].state_dict()}
                    torch.save(state, mdlParams['saveDir'] + '/checkpoint_best.pt')
                    
                duration = time.time() - start_time
                print(f"Fold {cv} | Epoch {step}/{mdlParams['training_steps']} | Loss: {loss:.4f} | Acc: {accuracy:.4f} | WACC: {np.mean(waccuracy):.4f}")

        print(f"Fold {cv} Training Complete.")
        modelVars.clear()