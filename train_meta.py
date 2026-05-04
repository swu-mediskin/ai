import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import sklearn.preprocessing
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

    mdlParams = {}   

    # 설정 파일 로드
    pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    mdlParams['trainSetState'] = 'train'
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]

    # 디바이스 설정
    if 'mps' in sys.argv[3]:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using {device}")
    elif 'gpu' in sys.argv[3] or 'cuda' in sys.argv[3]:
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if len(sys.argv) > 4:
        mdlParams['cv_subset'] = [int(s) for s in re.findall(r'\d+',sys.argv[4])]
        print("Training validation sets",mdlParams['cv_subset'])

    # Fold별 학습 시작
    cv_set = mdlParams.get('cv_subset', range(mdlParams['numCV']))
    
    for cv in cv_set:
        wandb.init(
            project="mediskin", 
            name=f"meta_only_all_real_same_fold{cv}", 
            config=mdlParams 
        )    

        print(f"\n--- Starting Meta-Only Fold {cv} ---")
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/meta_only_same_results'
        os.makedirs(mdlParams['saveDir'], exist_ok=True)

        importlib.reload(models)
        modelVars = {}
        modelVars['device'] = device

        mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
        eval_set = 'valInd'

        # Meta 데이터 스케일링
        if mdlParams.get('meta_features', None) is not None and mdlParams['scale_features']:
            mdlParams['feature_scaler_meta'] = sklearn.preprocessing.StandardScaler().fit(mdlParams['meta_array'][mdlParams['trainInd'],:])  
            print("Meta scaler fitted.")

        # 데이터로더 설정
        num_workers = psutil.cpu_count(logical=False)
        dataset_train = utils.PADDataset(mdlParams, 'trainInd')
        dataset_val = utils.PADDataset(mdlParams, 'valInd')
        
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, drop_last=True)
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=num_workers)

        modelVars['model'] = models.MetaOnlyModel(mdlParams).to(device)
        
        weights = mdlParams['class_weights'].to(device)
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=weights)

        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate_meta'])

        modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))
        modelVars['softmax'] = nn.Softmax(dim=1)

        mdlParams['valBest'] = 0.0
        start_time = time.time()
        
        for epoch in range(1, mdlParams['epoch'] + 1):
            pbar = tqdm(enumerate(modelVars['dataloader_trainInd']), 
                total=len(modelVars['dataloader_trainInd']), 
                desc=f"Fold {cv} | Epoch {epoch}")
            
            modelVars['model'].train()
            
            for j, (inputs, labels, indices) in pbar:
                meta_data = inputs[1].to(device)
                labels = labels.to(device)
                
                modelVars['optimizer'].zero_grad()
                outputs = modelVars['model'](meta_data)

                loss = modelVars['criterion'](outputs, labels)
                loss.backward()
                modelVars['optimizer'].step()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
            if epoch >= mdlParams['lowerLRat'] - mdlParams['lowerLRAfter']:
                modelVars['scheduler'].step()

            # 검증 단계
            modelVars['model'].eval()
            res = utils.getErrClassification_mgpu_meta(mdlParams, eval_set, modelVars)
            val_loss, accuracy, recall, precision, specificity, conf_matrix, f1, auc, predictions, targets, _ = res
                
            class_names = ['NEV', 'BCC', 'SCC', 'MEL']

            mean_recall = np.mean(recall)
            mean_precision = np.mean(precision)
            mean_specificity = np.mean(specificity)
            mean_f1 = np.mean(f1)

            metrics_dict = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "accuracy": accuracy,
                "mean_f1_score": mean_f1,
                "mean_auc": np.mean(auc),
                "mean_recall": mean_recall,
                "mean_precision": mean_precision,
                "mean_specificity": mean_specificity,
                "learning_rate": modelVars['optimizer'].param_groups[0]['lr']
            }

            for i, name in enumerate(class_names):
                metrics_dict[f"label_metrics/{name}_auc"] = auc[i]
                metrics_dict[f"label_metrics/{name}_recall"] = recall[i]      
                metrics_dict[f"label_metrics/{name}_precision"] = precision[i]
                metrics_dict[f"label_metrics/{name}_specificity"] = specificity[i]
                metrics_dict[f"label_metrics/{name}_f1_score"] = f1[i]

            metrics_dict["confusion_matrix"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.argmax(targets, axis=1) if targets.ndim > 1 else targets,
                preds=np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions,
                class_names=class_names
            )

            metrics_dict["pr_curve"] = wandb.plot.pr_curve(
                np.argmax(targets, axis=1) if targets.ndim > 1 else targets, 
                predictions, 
                labels=class_names
            )
                
            metrics_dict["roc_curve"] = wandb.plot.roc_curve(
                np.argmax(targets, axis=1) if targets.ndim > 1 else targets, 
                predictions, 
                labels=class_names
            )

            wandb.log(metrics_dict)
                
            print(f"Fold {cv} | Epoch {epoch} | Train Loss: {loss:.4f} | Acc: {accuracy:.4f} | Recall: {mean_recall:.4f} | Precision: {mean_precision:.4f}")

            # Best Model 저장 로직
            if mdlParams['valBest'] < mean_recall:
                mdlParams['valBest'] = mean_recall
                state = {'epoch': epoch, 'state_dict': modelVars['model'].state_dict(), 'optimizer': modelVars['optimizer'].state_dict()}
                torch.save(state, mdlParams['saveDir'] + '/best_fold' + str(cv) + '.pt')
                print("🌟 Best Model Saved!")


            duration = time.time() - start_time
            print(f"Fold {cv} | Epoch {epoch}/{mdlParams['epoch']} | "
                    f"Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f} | F1-score: {mean_f1:.4f} | "
                    f"Recall: {mean_recall:.4f} | Precision: {mean_precision:.4f} | Specificity: {mean_specificity:.4f} ")

        print(f"Fold {cv} Training Complete.")
        wandb.finish()

    modelVars.clear()
    