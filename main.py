import os
import io
import sys
import torch
import torch.nn as nn
import numpy as np
import sklearn.preprocessing
import importlib
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from torchvision import transforms
import models
from inference import build_meta_vector, build_text_tokens  # 추가

app = FastAPI(title="Mediskin AI API")

PC_CFG = "pad_cfg"
MODEL_CFG = "2019.pad_multi_all_real_aug"
DEVICE_STR = "mps"

pc_cfg = importlib.import_module(f'pc_cfgs.{PC_CFG}')
mdlParams = pc_cfg.mdlParams
model_cfg = importlib.import_module(f'cfgs.{MODEL_CFG}')
mdlParams.update(model_cfg.init(mdlParams))

if DEVICE_STR == 'mps' and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if mdlParams['model_type'] == 'biomedclip':
    model = models.BiomedCLIPMultiModal(mdlParams)
    tokenizer = models.get_biomedclip_tokenizer()
else:
    model = models.getModel(mdlParams)()
    if 'efficient' in mdlParams['model_type']:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, mdlParams['numClasses'])
    if mdlParams.get('meta_features', None) is not None:
        model = models.modify_meta(mdlParams, model)

default_chkpt = ('results/2019.pad_biomedclip/multi_all_real_aug_2/best_fold0.pt'
                  if mdlParams['model_type'] == 'biomedclip'
                  else 'results/2019.pad_multi_all_real_aug/multi_all_real_aug/best_fold0.pt')
chkPath = os.environ.get('MODEL_WEIGHTS_PATH', default_chkpt)
state = torch.load(chkPath, map_location=device, weights_only=False)
model.load_state_dict(state['state_dict'])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(mdlParams['input_size'][:2]),
    transforms.ToTensor(),
    transforms.Normalize(mean=mdlParams['setMean'], std=mdlParams['setStd'])
])

if mdlParams['model_type'] != 'biomedclip' and mdlParams['scale_features']:
    train_indices = mdlParams['trainIndCV'][0]
    scaler = sklearn.preprocessing.StandardScaler().fit(mdlParams['meta_array'][train_indices, :])
else:
    scaler = None

class_names = ['NEV', 'BCC', 'SCC', 'MEL']


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(0.0),
    gender: str = Form("unk"),
    region: str = Form("unk"),
    fitspatrick: float = Form(0.0),
    diameter_1: float = Form(0.0),
    diameter_2: float = Form(0.0),
    itch: str = Form("unk"),
    grew: str = Form("unk"),
    hurt: str = Form("unk"),
    changed: str = Form("unk"),
    bleed: str = Form("unk"),
    elevation: str = Form("unk"),
    smoke: str = Form("unk"),
    drink: str = Form("unk"),
    pesticide: str = Form("unk"),
    skin_cancer_history: str = Form("unk"),
    cancer_history: str = Form("unk"),
):
    try:
        image_data = await file.read()
        input_img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = preprocess(input_img).unsqueeze(0).to(device)

        meta_dict = {
            'age': age, 'gender': gender, 'region': region,
            'fitspatrick': fitspatrick, 'diameter_1': diameter_1, 'diameter_2': diameter_2,
            'itch': itch, 'grew': grew, 'hurt': hurt, 'changed': changed,
            'bleed': bleed, 'elevation': elevation, 'smoke': smoke, 'drink': drink,
            'pesticide': pesticide, 'skin_cancer_history': skin_cancer_history,
            'cancer_history': cancer_history
        }

        if mdlParams['model_type'] == 'biomedclip':
            meta_array = build_text_tokens(meta_dict, mdlParams, tokenizer)
        else:
            meta_array = build_meta_vector(meta_dict, mdlParams, scaler)
        meta_tensor = torch.from_numpy(meta_array).float().to(device)

        with torch.no_grad():
            outputs = model([img_tensor, meta_tensor])
            probs = nn.Softmax(dim=1)(outputs)
            conf, pred = torch.max(probs, 1)

        return {
            "prediction": class_names[pred.item()],
            "confidence": float(conf.item()),
            "all_probabilities": probs.cpu().numpy().tolist()[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)