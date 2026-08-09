import os
import sys
import importlib
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

import models


class GradCAM:
    """target_layer에 forward/backward hook을 걸어 Grad-CAM을 계산하는 범용 클래스."""

    def __init__(self, target_layer, reshape_transform=None):
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        act = output
        if self.reshape_transform is not None:
            act = self.reshape_transform(act)
        self.activations = act.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = grad.detach()

    def generate(self, forward_fn, class_idx=None):
        """forward_fn: () -> logits (1, numClasses) 텐서를 반환하는 클로저."""
        logits = forward_fn()
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1))  # (1, H, W)
        cam = cam.squeeze(0).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx


def vit_reshape_transform(tensor, grid_size=14):
    """(B, 1+N, C) -> CLS 제거 후 (B, C, grid, grid)로 재배열."""
    b, n, c = tensor.shape
    patches = tensor[:, 1:, :]  # CLS 토큰 버림
    patches = patches.reshape(b, grid_size, grid_size, c)
    return patches.permute(0, 3, 1, 2)  # (B, C, H, W)


def get_target_layer(model, model_type):
    if model_type == 'biomedclip':
        return model.clip.visual.trunk.blocks[-1].norm1, vit_reshape_transform
    elif 'efficient' in model_type:
        return model._conv_head, None
    else:
        raise ValueError(f"Grad-CAM 대상으로 지원하지 않는 model_type: {model_type}")


def build_model(mdlParams, device):
    if mdlParams['model_type'] == 'biomedclip':
        model = models.BiomedCLIPMultiModal(mdlParams).to(device)
    else:
        model = models.getModel(mdlParams)()
        if 'efficient' in mdlParams['model_type']:
            num_ftrs = model._fc.in_features
            model._fc = torch.nn.Linear(num_ftrs, mdlParams['numClasses'])
        model = model.to(device)
        if mdlParams.get('meta_features', None) is not None:
            model = models.modify_meta(mdlParams, model)
            model = model.to(device)
    return model


def load_checkpoint(model, mdlParams, cv, device, which='best'):
    chk_path = os.path.join(mdlParams['saveDir'], f'{which}_fold{cv}.pt')
    state = torch.load(chk_path, map_location=device, weights_only=False)
    model.load_state_dict(state['state_dict'])
    return model


def preprocess_image(img_path, mdlParams, device):
    """PADDataset의 valInd 전처리(CenterCrop(1.5x input_size) -> Resize)와 동일하게 맞춤."""
    input_img = Image.open(img_path).convert('RGB')
    input_size = mdlParams['input_size'][:2]
    if input_img.size[0] < input_size[0] or input_img.size[1] < input_size[1]:
        input_img = input_img.resize(tuple(input_size), Image.BILINEAR)
    tfm = transforms.Compose([
        transforms.CenterCrop(np.int32(input_size[0] * 1.5)),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mdlParams['setMean'], std=mdlParams['setStd']),
    ])
    tensor = tfm(input_img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    cropped = transforms.CenterCrop(np.int32(input_size[0] * 1.5))(input_img)
    orig_resized = np.array(cropped.resize(tuple(input_size)))
    return tensor, orig_resized


def run_gradcam_for_image(mdlParams, model, device, img_path, second_input, out_path, class_idx=None):
    """second_input: BiomedCLIP이면 텍스트 토큰 텐서(1, context_len), EfficientNet+meta면 메타 벡터 텐서."""
    model.eval()
    img_tensor, orig_resized = preprocess_image(img_path, mdlParams, device)

    target_layer, reshape_fn = get_target_layer(model, mdlParams['model_type'])
    cam_obj = GradCAM(target_layer, reshape_transform=reshape_fn)

    second_input = second_input.to(device)
    if mdlParams['model_type'] == 'biomedclip':
        forward_fn = lambda: model([img_tensor, second_input], gradcam=True)
    else:
        forward_fn = lambda: model([img_tensor, second_input])

    with torch.enable_grad():
        model.zero_grad()
        cam, pred_idx = cam_obj.generate(forward_fn, class_idx=class_idx)

    cam_resized = cv2.resize(cam, (mdlParams['input_size'][1], mdlParams['input_size'][0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * orig_resized + 0.4 * heatmap).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return pred_idx


if __name__ == '__main__':
    model_cfg_arg = sys.argv[2]
    pc_cfg = importlib.import_module('pc_cfgs.' + sys.argv[1])
    mdlParams = {}
    mdlParams.update(pc_cfg.mdlParams)
    model_cfg = importlib.import_module('cfgs.' + model_cfg_arg)
    mdlParams.update(model_cfg.init(mdlParams))
    mdlParams['saveDir'] = mdlParams['saveDir'] + model_cfg_arg + '/' + mdlParams['mdlName']

    device_str = sys.argv[3]
    device = torch.device('mps' if 'mps' in device_str else ('cuda:0' if 'gpu' in device_str or 'cuda' in device_str else 'cpu'))
    which = sys.argv[4] if len(sys.argv) > 4 else 'best'
    cv = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    target = sys.argv[6] if len(sys.argv) > 6 else 'val'
    n = int(sys.argv[7]) if len(sys.argv) > 7 else 8

    class_names = ['NEV', 'BCC', 'SCC', 'MEL']

    model = build_model(mdlParams, device)
    model = load_checkpoint(model, mdlParams, cv, device, which=which)

    if target == 'val':
        val_indices = mdlParams['valIndCV'][cv][:n]
        img_paths = [mdlParams['im_paths'][i] for i in val_indices]
    else:
        img_paths = [target]

    for img_path in img_paths:
        second_input = torch.from_numpy(mdlParams['meta_dict'][img_path]).unsqueeze(0)
        out_path = os.path.join(mdlParams['saveDir'], 'gradcam', f'fold{cv}', os.path.basename(img_path))
        pred_idx = run_gradcam_for_image(mdlParams, model, device, img_path, second_input, out_path)
        print(f"{img_path} -> pred={class_names[pred_idx]} saved to {out_path}")
