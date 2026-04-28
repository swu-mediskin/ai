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
import sklearn.preprocessing
import os
import sys
from glob import glob
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

    pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    mdlParams['trainSetState'] = 'train'
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]

    # gpu 환경 설정
    if 'mps' in sys.argv[3]:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
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
    if mdlParams.get('cv_subset',None) is not None:
        cv_set = mdlParams['cv_subset']
    else:
        cv_set = range(mdlParams['numCV'])
    for cv in cv_set:
        wandb.init(
        project="mediskin", 
        name=f"meta_all_real_norm_fold{cv}", 
        config=mdlParams 
        )    

        print(f"\n--- Starting Fold {cv} ---")
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/meta_all_real_norm'
        if not os.path.isdir(mdlParams['saveDirBase']): os.makedirs(mdlParams['saveDirBase'], exist_ok=True)
        if not os.path.isdir(mdlParams['saveDir']): os.makedirs(mdlParams['saveDir'], exist_ok=True)

        importlib.reload(models)
        modelVars = {}
        modelVars['device'] = device
        print(modelVars['device'])

        mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
        eval_set = 'valInd'

        if mdlParams['subtract_set_mean'] == 1:
            mdlParams['setMean'] = np.mean(mdlParams['images_means'][mdlParams['trainInd'],:],(0))
            print("Set Mean",mdlParams['setMean'])

        # Meta scaler
        if mdlParams.get('meta_features',None) is not None and mdlParams['scale_features']:
            mdlParams['feature_scaler_meta'] = sklearn.preprocessing.StandardScaler().fit(mdlParams['meta_array'][mdlParams['trainInd'],:])  
            print("scaler mean",mdlParams['feature_scaler_meta'].mean_,"var",mdlParams['feature_scaler_meta'].var_)  

        num_workers = psutil.cpu_count(logical=False)
        dataset_train = utils.PADDataset(mdlParams, 'trainInd')
        dataset_val = utils.PADDataset(mdlParams, 'valInd')
        
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=False)

        modelVars['model'] = models.getModel(mdlParams)()
        modelVars['softmax'] = nn.Softmax(dim=1)
        
        # 학습된 가중치 불러오는 로직
        if mdlParams.get('meta_features', None) is not None:
            chkPath = os.path.join(mdlParams['model_load_path'], f'best_fold{cv}.pt')
            
            if os.path.isfile(chkPath):
                print(f"✅ Restoring best lesion-trained CNN from: {chkPath}")
                
                state = torch.load(chkPath, map_location=device)
                
                curr_model_dict = modelVars['model'].state_dict()
            
                for name, param in state['state_dict'].items():
                    if name in curr_model_dict and curr_model_dict[name].shape == param.shape:
                        curr_model_dict[name].copy_(param)
                    else:
                        if name not in curr_model_dict:
                            print(f"⚠️ Skipping {name}: not in current model")
                
                modelVars['model'].load_state_dict(curr_model_dict)
            else:
                print(f"❌ Warning: No best model found at {chkPath}. Starting from scratch.")

        if 'Dense' in mdlParams['model_type']:
            if mdlParams['input_size'][0] != 224:
                modelVars['model'] = utils.modify_densenet_avg_pool(modelVars['model'])
                #print(modelVars['model'])
            num_ftrs = modelVars['model'].classifier.in_features
            modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
            #print(modelVars['model'])
        elif 'efficient' in mdlParams['model_type']:
            num_ftrs = modelVars['model']._fc.in_features
            modelVars['model']._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
        
        modelVars['model'] = modelVars['model'].to(device)

        if mdlParams.get('meta_features',None) is not None:
            # freeze cnn first
            if mdlParams['freeze_cnn']:
                # deactivate all
                for param in modelVars['model'].parameters():
                    param.requires_grad = False            
                if 'efficient' in mdlParams['model_type']:
                    # Activate fc
                    for param in modelVars['model']._fc.parameters():
                        param.requires_grad = True                              
            else:
                # mark cnn parameters
                for param in modelVars['model'].parameters():
                    param.is_cnn_param = True
                # unmark fc
                for param in modelVars['model']._fc.parameters():
                    param.is_cnn_param = False                              
            # modify model
            modelVars['model'] = models.modify_meta(mdlParams,modelVars['model']) 
            modelVars['model'] = modelVars['model'].to(device) 
            # Mark new parameters
            for param in modelVars['model'].parameters():
                if not hasattr(param, 'is_cnn_param'):
                    param.is_cnn_param = False        

        
        weights = mdlParams['class_weights'].to(device)
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=weights)

        if mdlParams.get('meta_features',None) is not None:
            if mdlParams['freeze_cnn']:
                modelVars['optimizer'] = optim.Adam(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate_meta'])
                # sanity check
                for param in filter(lambda p: p.requires_grad, modelVars['model'].parameters()):
                    print(param.name,param.shape)
            else:
                modelVars['optimizer'] = optim.Adam([
                                                    {'params': filter(lambda p: not p.is_cnn_param, modelVars['model'].parameters()), 'lr': mdlParams['learning_rate_meta']},
                                                    {'params': filter(lambda p: p.is_cnn_param, modelVars['model'].parameters()), 'lr': mdlParams['learning_rate']}
                                                    ], lr=mdlParams['learning_rate'])
        else:
            modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])


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
                if mdlParams.get('meta_features',None) is not None: 
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)
                else:
                    inputs = inputs.to(device)
                
                labels = labels.to(device)
                
                modelVars['optimizer'].zero_grad()
                outputs = modelVars['model'](inputs)

                loss = modelVars['criterion'](outputs, labels)
                loss.backward()
                modelVars['optimizer'].step()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
            if epoch >= mdlParams['lowerLRat'] - mdlParams['lowerLRAfter']:
                modelVars['scheduler'].step()

            modelVars['model'].eval()
            res = utils.getErrClassification_mgpu(mdlParams, eval_set, modelVars)
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
    