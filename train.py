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

    mdlParams = {}   

    pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    mdlParams['trainSetState'] = 'train'
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]

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

    if len(sys.argv) > 4:
        cv_set = [int(s) for s in re.findall(r'\d+', sys.argv[4])]
    else:
        cv_set = range(mdlParams['numCV'])

    prevFile = Path(mdlParams['saveDirBase'] + '/CV.pkl')
    allData = pickle.load(open(prevFile, 'rb')) if prevFile.exists() else {
        k: {} for k in ['f1Best', 'sensBest', 'specBest', 'accBest', 'waccBest', 'aucBest', 'convergeTime', 'bestPred', 'targets']
    }

    for cv in cv_set:
        wandb.init(
        project="mediskin", 
        name=f"image-only-baseline-fold{cv}", 
        config=mdlParams 
        )    

        print(f"\n--- Starting Fold {cv} ---")
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
        if not os.path.isdir(mdlParams['saveDirBase']): os.makedirs(mdlParams['saveDirBase'], exist_ok=True)
        if not os.path.isdir(mdlParams['saveDir']): os.makedirs(mdlParams['saveDir'], exist_ok=True)

        mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
        eval_set = 'valInd'

        importlib.reload(models)
        modelVars = {'device': device}
        modelVars['model'] = models.getModel(mdlParams)()
        modelVars['softmax'] = nn.Softmax(dim=1)


        if 'efficient' in mdlParams['model_type']:
            num_ftrs = modelVars['model']._fc.in_features
            modelVars['model']._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
        
        modelVars['model'] = modelVars['model'].to(device)

        num_workers = psutil.cpu_count(logical=False)
        dataset_train = utils.ISICDataset(mdlParams, 'trainInd')
        dataset_val = utils.ISICDataset(mdlParams, 'valInd')
        
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=False)

        modelVars['criterion'] = nn.CrossEntropyLoss()
        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])
        modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/mdlParams['LRstep'])

        mdlParams['valBest'] = 0.0
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
                "train_loss": train_loss.item(),
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
                
            print(f"Fold {cv} | Epoch {epoch} | Train Loss: {train_loss:.4f} | Acc: {accuracy:.4f} | Recall: {mean_recall:.4f} | Precision: {mean_precision:.4f}")

            # Best Model 저장 로직
            if mdlParams['valBest'] < mean_recall:
                mdlParams['valBest'] = mean_recall
                state = {'epoch': epoch, 'state_dict': modelVars['model'].state_dict(), 'optimizer': modelVars['optimizer'].state_dict()}
                torch.save(state, mdlParams['saveDir'] + '/checkpoint_best.pt')
                print("🌟 Best Model Saved!")


            duration = time.time() - start_time
            print(f"Fold {cv} | Epoch {epoch}/{mdlParams['epoch']} | "
                    f"Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f} | F1-score: {mean_f1:.4f} | "
                    f"Recall: {mean_recall:.4f} | Precision: {mean_precision:.4f} | Specificity: {mean_specificity:.4f} ")

        print(f"Fold {cv} Training Complete.")
        wandb.finish()

    modelVars.clear()
    