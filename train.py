import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
from scipy import io
import pickle
from pathlib import Path
import os
import sys
import re
import importlib
import time
import utils
import psutil
import models
import wandb
from tqdm import tqdm


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
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("MPS not available. Falling back to CPU.")
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
        wandb.init(
        project="mediskin", # 프로젝트 이름
        name=f"image-only-baseline-fold{cv}", # 실험 이름
        config=mdlParams # 하이퍼파라미터 기록
        )    

        print(f"\n--- Starting Fold {cv} ---")
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
        if not os.path.isdir(mdlParams['saveDirBase']): os.makedirs(mdlParams['saveDirBase'], exist_ok=True)
        if not os.path.isdir(mdlParams['saveDir']): os.makedirs(mdlParams['saveDir'], exist_ok=True)

        # 인덱스 설정 
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
        
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=False)

        # 6. 손실 함수 및 옵티마이저
        # 4개 클래스 단순 분류이므로 CrossEntropy 사용
        modelVars['criterion'] = nn.CrossEntropyLoss()
        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])
        modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/mdlParams['LRstep'])

        # 7. 학습 루프
        mdlParams['valBest'] = 1000
        start_time = time.time()
        for epoch in range(1, mdlParams['epoch'] + 1):
            pbar = tqdm(enumerate(modelVars['dataloader_trainInd']), 
                total=len(modelVars['dataloader_trainInd']), 
                desc=f"Fold {cv} | Epoch {epoch}")
            modelVars['model'].train()
            
            for j, (inputs, labels, indices) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)    
                modelVars['optimizer'].zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = modelVars['model'](inputs)
                    train_loss = modelVars['criterion'](outputs, labels)
                    train_loss.backward()
                    modelVars['optimizer'].step()
                    
                if j % 10 == 0:
                    pbar.set_postfix(loss=f"{train_loss.item():.4f}")
        
            # 스케줄러 업데이트
            if epoch >= mdlParams['lowerLRat'] - mdlParams['lowerLRAfter']:
                modelVars['scheduler'].step()

            # 평가 및 저장
            modelVars['model'].eval()
            res = utils.getErrClassification_mgpu(mdlParams, eval_set, modelVars)
            val_loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = res
                
            class_names = ['NEV', 'BCC', 'SCC', 'MEL']
                
            # --- Precision 계산 (Confusion Matrix 활용) ---
            # Precision = TP / (TP + FP)
            precision = np.zeros(mdlParams['numClasses'])
            for k in range(mdlParams['numClasses']):
                tp = conf_matrix[k, k]
                fp = np.sum(conf_matrix[:, k]) - tp
                precision[k] = tp / (tp + fp) if (tp + fp) > 0 else 0
                
            # --- 평균 지표 계산 ---
            mean_recall = np.mean(sensitivity)
            mean_precision = np.mean(precision)
            mean_sensitivity = np.mean(sensitivity)
            mean_specificity = np.mean(specificity)

            metrics_dict = {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "val_loss": val_loss,
                "accuracy": accuracy,
                "f1_score": f1,
                "mean_auc": np.mean(auc),
                "mean_wacc": np.mean(waccuracy),
                "mean_recall": mean_recall,
                "mean_precision": mean_precision,
                "mean_sensitivity": mean_sensitivity,
                "mean_specificity": mean_specificity,
                "learning_rate": modelVars['optimizer'].param_groups[0]['lr']
            }

            # 2. 클래스별 개별 지표 추가
            for i, name in enumerate(class_names):
                metrics_dict[f"label_metrics/{name}_auc"] = auc[i]
                metrics_dict[f"label_metrics/{name}_recall"] = sensitivity[i]      # Recall 추가
                metrics_dict[f"label_metrics/{name}_precision"] = precision[i]    # Precision 추가
                metrics_dict[f"label_metrics/{name}_sens"] = sensitivity[i]
                metrics_dict[f"label_metrics/{name}_spec"] = specificity[i]

            # 3. Confusion Matrix 시각화
            metrics_dict["confusion_matrix"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.argmax(targets, axis=1) if targets.ndim > 1 else targets,
                preds=np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions,
                class_names=class_names
            )

            # 4. PR Curve 추가 (Precision-Recall 곡선)
            metrics_dict["pr_curve"] = wandb.plot.pr_curve(
                np.argmax(targets, axis=1) if targets.ndim > 1 else targets, 
                predictions, 
                labels=class_names
            )
                
            # ROC Curve
            metrics_dict["roc_curve"] = wandb.plot.roc_curve(
                np.argmax(targets, axis=1) if targets.ndim > 1 else targets, 
                predictions, 
                labels=class_names
            )

            # 모든 정보 로그 전송
            wandb.log(metrics_dict)
                
            # 터미널 출력용 (현재 상황 확인)
            print(f"Fold {cv} | Epoch {epoch} | Train Loss: {train_loss:.4f} | Acc: {accuracy:.4f} | Recall: {mean_recall:.4f} | Precision: {mean_precision:.4f}")

            eval_metric = -np.mean(waccuracy)
            if eval_metric < mdlParams['valBest']:
                mdlParams['valBest'] = eval_metric
                # 최적 모델 저장 로직 (생략 가능하나 유지는 권장)
                state = {'epoch': epoch, 'state_dict': modelVars['model'].state_dict(), 'optimizer': modelVars['optimizer'].state_dict()}
                torch.save(state, mdlParams['saveDir'] + '/checkpoint_best.pt')
                print("🌟 Best Model Saved!")
                    
            # 시간 계산 및 터미널 출력
            duration = time.time() - start_time
            print(f"Fold {cv} | Epoch {epoch}/{mdlParams['epoch']} | "
                    f"Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f} | WACC: {np.mean(waccuracy):.4f} | "
                    f"Recall: {mean_recall:.4f} | Precision: {mean_precision:.4f}")

        print(f"Fold {cv} Training Complete.")
        wandb.finish()

    modelVars.clear()
    