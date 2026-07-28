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


def run_fold(mdlParams, device, cv):
    """wandb.init()과 saveDir 세팅은 호출부(caller)에서 끝냈다고 가정하고, 학습만 담당."""
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

    modelVars['softmax'] = nn.Softmax(dim=1)

    if mdlParams['model_type'] == 'biomedclip':
        modelVars['model'] = models.BiomedCLIPMultiModal(mdlParams).to(device)
    else:
        modelVars['model'] = models.getModel(mdlParams)()

        # 학습된 가중치 불러오는 로직
        if mdlParams.get('meta_features', None) is not None:
            chkPath = os.path.join(mdlParams['model_load_path'], f'best_fold{cv}.pt')

            if os.path.isfile(chkPath):
                print(f"✅ Restoring best lesion-trained CNN from: {chkPath}")

                state = torch.load(chkPath, map_location=device, weights_only=False)

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
            num_ftrs = modelVars['model'].classifier.in_features
            modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
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

    if mdlParams['model_type'] == 'biomedclip':
        use_lora = mdlParams.get('use_lora_visual', False) or mdlParams.get('use_lora_text', False)
        use_unfreeze = mdlParams.get('unfreeze_visual_blocks', 0) > 0 or mdlParams.get('unfreeze_text_blocks', 0) > 0
        if use_lora or use_unfreeze:
            param_groups = [
                {'params': filter(lambda p: p.requires_grad and not getattr(p, 'is_cnn_param', False) and not getattr(p, 'is_lora_param', False), modelVars['model'].parameters()), 'lr': mdlParams['learning_rate_meta']},
            ]
            if use_unfreeze:
                param_groups.append({'params': filter(lambda p: p.requires_grad and getattr(p, 'is_cnn_param', False), modelVars['model'].parameters()), 'lr': mdlParams['learning_rate']})
            if use_lora:
                param_groups.append({'params': filter(lambda p: p.requires_grad and getattr(p, 'is_lora_param', False), modelVars['model'].parameters()), 'lr': mdlParams.get('learning_rate_lora', 1e-4)})
            modelVars['optimizer'] = optim.AdamW(param_groups, lr=mdlParams['learning_rate'], weight_decay=0.2, betas=(0.9, 0.98), eps=1.0e-6)
        else:
            modelVars['optimizer'] = optim.AdamW(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate_meta'], weight_decay=0.2, betas=(0.9, 0.98), eps=1.0e-6)
    elif mdlParams.get('meta_features',None) is not None:
        if mdlParams['freeze_cnn']:
            modelVars['optimizer'] = optim.AdamW(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate_meta'], weight_decay=0.2, betas=(0.9, 0.98), eps=1.0e-6)
            # sanity check
            for param in filter(lambda p: p.requires_grad, modelVars['model'].parameters()):
                print(param.name,param.shape)
        else:
            modelVars['optimizer'] = optim.AdamW([
                                                {'params': filter(lambda p: not p.is_cnn_param, modelVars['model'].parameters()), 'lr': mdlParams['learning_rate_meta']},
                                                {'params': filter(lambda p: p.is_cnn_param, modelVars['model'].parameters()), 'lr': mdlParams['learning_rate']}
                                                ], lr=mdlParams['learning_rate'], weight_decay=0.2, betas=(0.9, 0.98), eps=1.0e-6)
    else:
        modelVars['optimizer'] = optim.AdamW(modelVars['model'].parameters(), lr=mdlParams['learning_rate'], weight_decay=0.2, betas=(0.9, 0.98), eps=1.0e-6)


    # 200 step 선형 warmup 후 나머지 스텝 동안 cosine decay (step 단위 스케줄링).
    steps_per_epoch = len(modelVars['dataloader_trainInd'])
    total_steps = mdlParams['epoch'] * steps_per_epoch
    warmup_steps = min(200, total_steps)
    warmup_scheduler = lr_scheduler.LinearLR(modelVars['optimizer'], start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(modelVars['optimizer'], T_max=max(total_steps - warmup_steps, 1))
    modelVars['scheduler'] = lr_scheduler.SequentialLR(modelVars['optimizer'], schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    modelVars['softmax'] = nn.Softmax(dim=1)


    mdlParams['valBest'] = 0.0
    start_time = time.time()
    for epoch in range(1, mdlParams['epoch'] + 1):
        pbar = tqdm(enumerate(modelVars['dataloader_trainInd']),
            total=len(modelVars['dataloader_trainInd']),
            desc=f"Fold {cv} | Epoch {epoch}")
        modelVars['model'].train()

        class_train_loss_sum = torch.zeros(mdlParams['numClasses']).to(device)
        class_train_loss_count = torch.zeros(mdlParams['numClasses']).to(device)

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
            modelVars['scheduler'].step()

            with torch.no_grad():
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                for c in range(mdlParams['numClasses']):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        class_train_loss_sum[c] += per_sample_loss[mask].sum()
                        class_train_loss_count[c] += mask.sum()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        modelVars['model'].eval()
        res = utils.getErrClassification_mgpu(mdlParams, eval_set, modelVars)
        val_loss, accuracy, recall, precision, specificity, conf_matrix, f1, auc, auprc, predictions, targets, _, class_val_loss = res

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
            "mean_auprc": np.mean(auprc),
            "mean_recall": mean_recall,
            "mean_precision": mean_precision,
            "mean_specificity": mean_specificity,
            "learning_rate": modelVars['optimizer'].param_groups[0]['lr']
        }

        class_train_losses = class_train_loss_sum / (class_train_loss_count + 1e-8)

        for i, name in enumerate(class_names):
            metrics_dict[f"label_metrics/{name}_val_loss"] = class_val_loss[i]
            metrics_dict[f"label_metrics/{name}_train_loss"] = class_train_losses[i].item()
            metrics_dict[f"label_metrics/{name}_auc"] = auc[i]
            metrics_dict[f"label_metrics/{name}_auprc"] = auprc[i]
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
    modelVars.clear()


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
            name=mdlParams['mdlName'] + "_f" + str(cv),
            config=mdlParams
        )

        print(f"\n--- Starting Fold {cv} ---")
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/' + mdlParams['mdlName']
        if not os.path.isdir(mdlParams['saveDirBase']): os.makedirs(mdlParams['saveDirBase'], exist_ok=True)
        if not os.path.isdir(mdlParams['saveDir']): os.makedirs(mdlParams['saveDir'], exist_ok=True)

        run_fold(mdlParams, device, cv)

        wandb.finish()
