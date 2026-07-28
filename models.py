import torch
import numbers
import numpy as np
import functools
import h5py
import math
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F
import types
import torch
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import torch.nn as nn
import open_clip
from peft import LoraConfig, inject_adapter_in_model

BIOMEDCLIP_TAG = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
BIOMEDCLIP_CONTEXT_LENGTH = 256
BIOMEDCLIP_EMBED_DIM = 512

def Dense121(config):
    return models.densenet121(pretrained=True)

def Dense161(config):
    return models.densenet169(pretrained=True)

def Dense169(config):
    return models.densenet161(pretrained=True)

def Dense201(config):
    return models.densenet201(pretrained=True)

def Resnet50(config):
    return pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

def Resnet101(config):
    return models.resnet101(pretrained=True)

def InceptionV3(config):
    return models.inception_v3(pretrained=True)

def se_resnext50(config):
    return pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')

def se_resnext101(config):
    return pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')

def se_resnet50(config):
    return pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')

def se_resnet101(config):
    return pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')

def se_resnet152(config):
    return pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet')

def resnext101(config):
    return pretrainedmodels.__dict__['resnext101_32x4d'](num_classes=1000, pretrained='imagenet')

def resnext101_64(config):
    return pretrainedmodels.__dict__['resnext101_64x4d'](num_classes=1000, pretrained='imagenet')

def senet154(config):
    return pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')

def polynet(config):
    return pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet')

def dpn92(config):
    return pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained='imagenet+5k')

def dpn68b(config):
    return pretrainedmodels.__dict__['dpn68b'](num_classes=1000, pretrained='imagenet+5k')

def nasnetamobile(config):
    return pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')

def resnext101_32_8_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

def resnext101_32_16_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

def resnext101_32_32_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')

def resnext101_32_48_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')

def efficientnet_b0(config):
    return EfficientNet.from_pretrained('efficientnet-b0',num_classes=config['numClasses'])

def efficientnet_b1(config):
    return EfficientNet.from_pretrained('efficientnet-b1',num_classes=config['numClasses'])

def efficientnet_b2(config):
    return EfficientNet.from_pretrained('efficientnet-b2',num_classes=config['numClasses'])

def efficientnet_b3(config):
    return EfficientNet.from_pretrained('efficientnet-b3',num_classes=config['numClasses'])

def efficientnet_b4(config):
    return EfficientNet.from_pretrained('efficientnet-b4',num_classes=config['numClasses'])

def efficientnet_b5(config):
    return EfficientNet.from_pretrained('efficientnet-b5',num_classes=config['numClasses'])       

def efficientnet_b6(config):
    return EfficientNet.from_pretrained('efficientnet-b6',num_classes=config['numClasses'])   

def efficientnet_b7(config):
    return EfficientNet.from_pretrained('efficientnet-b7',num_classes=config['numClasses'])  

def modify_meta(mdlParams,model):
    # Define FC layers
    if len(mdlParams['fc_layers_before']) > 1:
        model.meta_before = nn.Sequential(nn.Linear(mdlParams['meta_array'].shape[1],mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']),
                                    nn.Linear(mdlParams['fc_layers_before'][0],mdlParams['fc_layers_before'][1]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][1]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    else:
        model.meta_before = nn.Sequential(nn.Linear(mdlParams['meta_array'].shape[1],mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    # Define fc layers after
    if len(mdlParams['fc_layers_after']) > 0:
        if 'efficient' in mdlParams['model_type']:
            num_cnn_features = model._fc.in_features 
        elif 'wsl' in mdlParams['model_type']:
            num_cnn_features = model.fc.in_features  
        else:
            num_cnn_features = model.last_linear.in_features     
        model.meta_after = nn.Sequential(nn.Linear(mdlParams['fc_layers_before'][-1]+num_cnn_features,mdlParams['fc_layers_after'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_after'][0]),
                                    nn.ReLU())
        classifier_in_features = mdlParams['fc_layers_after'][0] 
    else:
        model.meta_after = None
        classifier_in_features = mdlParams['fc_layers_before'][-1]+model._fc.in_features
    # Modify classifier
    if 'efficient' in mdlParams['model_type']:
        model._fc = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    elif 'wsl' in mdlParams['model_type']:
        model.fc = nn.Linear(classifier_in_features, mdlParams['numClasses']) 
    else:
        model.last_linear = nn.Linear(classifier_in_features, mdlParams['numClasses'])       
    # Modify forward pass
    def new_forward(self, inputs):
        x, meta_data = inputs
        # Normal CNN features
        if 'efficient' in mdlParams['model_type']:
            # Convolution layers
            cnn_features = self.extract_features(x)
            # Pooling and final linear layer
            cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)
            # if self._dropout:
            #     cnn_features = F.dropout(cnn_features, p=self._dropout, training=self.training)
        elif 'wsl' in mdlParams['model_type']:
            cnn_features = self.conv1(x)
            cnn_features = self.bn1(cnn_features)
            cnn_features = self.relu(cnn_features)
            cnn_features = self.maxpool(cnn_features)

            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)

            cnn_features = self.avgpool(cnn_features)
            cnn_features = torch.flatten(cnn_features, 1) 
        else:
            cnn_features = self.layer0(x)
            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)   
            cnn_features = self.avg_pool(cnn_features)
            if self.dropout is not None:
                cnn_features = self.dropout(cnn_features)
            cnn_features = cnn_features.view(cnn_features.size(0), -1)                                
        # Meta part
        #print(meta_data.shape,meta_data)
        meta_features = self.meta_before(meta_data)

        # Cat
        features = torch.cat((cnn_features,meta_features),dim=1)
        #print("features cat",features.shape)
        if self.meta_after is not None:
            features = self.meta_after(features)
        # Classifier
        if 'efficient' in mdlParams['model_type']:
            output = self._fc(features)
        elif 'wsl' in mdlParams['model_type']:
            output = self.fc(features)
        else:
            output = self.last_linear(features)
        return output
    model.forward  = types.MethodType(new_forward, model)
    return model                                                                                                                       

model_map = OrderedDict([('Dense121',  Dense121),
                        ('Dense169' , Dense161),
                        ('Dense161' , Dense169),
                        ('Dense201' , Dense201),
                        ('Resnet50' , Resnet50),
                        ('Resnet101' , Resnet101),   
                        ('InceptionV3', InceptionV3),# models.inception_v3(pretrained=True),
                        ('se_resnext50', se_resnext50),
                        ('se_resnext101', se_resnext101),
                        ('se_resnet50', se_resnet50),
                        ('se_resnet101', se_resnet101),
                        ('se_resnet152', se_resnet152),
                        ('resnext101', resnext101),
                        ('resnext101_64', resnext101_64),
                        ('senet154', senet154),
                        ('polynet', polynet),
                        ('dpn92', dpn92),
                        ('dpn68b', dpn68b),
                        ('nasnetamobile', nasnetamobile),
                        ('resnext101_32_8_wsl', resnext101_32_8_wsl),
                        ('resnext101_32_16_wsl', resnext101_32_16_wsl),
                        ('resnext101_32_32_wsl', resnext101_32_32_wsl),
                        ('resnext101_32_48_wsl', resnext101_32_48_wsl),
                        ('efficientnet-b0', efficientnet_b0), 
                        ('efficientnet-b1', efficientnet_b1), 
                        ('efficientnet-b2', efficientnet_b2), 
                        ('efficientnet-b3', efficientnet_b3),  
                        ('efficientnet-b4', efficientnet_b4), 
                        ('efficientnet-b5', efficientnet_b5),  
                        ('efficientnet-b6', efficientnet_b6), 
                        ('efficientnet-b7', efficientnet_b7),
                    ])

def getModel(config):
  """Returns a function for a model
  Args:
    config: dictionary, contains configuration
  Returns:
    model: A class that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if config['model_type'] in model_map:
    func = model_map[config['model_type'] ]
    @functools.wraps(func)
    def model():
        return func(config)
  else:
      raise ValueError('Name of model unknown %s' % config['model_name'] )
  return model

import torch.nn as nn

class MetaOnlyModel(nn.Module):
    def __init__(self, mdlParams):
        super(MetaOnlyModel, self).__init__()
        # 메타데이터 피처 개수 (70차원 등)
        input_dim = mdlParams['meta_array'].shape[1] 
        
        layers = []
        curr_dim = input_dim
        
        # mdlParams['fc_layers_before'] 설정을 사용하여 층을 쌓음
        # 예: [128, 64] 라면 Linear(70, 128) -> Linear(128, 64)
        for h_dim in mdlParams['fc_layers_before']:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=mdlParams['dropout_meta']))
            curr_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_dim, mdlParams['numClasses'])

    def forward(self, x):
        # x는 메타데이터 텐서
        # 만약 inputs[1](meta)만 바로 넘긴다면 x를 그대로 쓰면 됨
        x = self.backbone(x)
        return self.classifier(x)


def get_biomedclip_tokenizer():
    return open_clip.get_tokenizer(BIOMEDCLIP_TAG)


def get_biomedclip_preprocess():
    _, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_TAG)
    return preprocess


def _lora_layer_range(num_total, num_unfreeze, num_lora):
    """LoRA 대상 블록 인덱스 구간을 계산한다.

    unfreeze로 이미 풀 파인튜닝되는 마지막 num_unfreeze개 블록과 겹치지 않도록,
    그 바로 앞의 num_lora개 블록에 LoRA를 적용한다 (num_unfreeze=0이면 기존과 동일하게
    마지막 num_lora개 블록). 예: 전체 12블록, unfreeze=2, lora=4 -> LoRA는 블록 [6,7,8,9],
    unfreeze는 블록 [10,11].
    """
    end = num_total - num_unfreeze
    start = end - num_lora
    if start < 0:
        raise ValueError(
            f"lora_blocks({num_lora}) + unfreeze_blocks({num_unfreeze})가 전체 블록 수"
            f"({num_total})보다 큽니다."
        )
    return list(range(start, end))


def _inject_lora(scope_module, block_attr_name, layer_idxs, target_modules, r, alpha, dropout):
    """scope_module 아래 <block_attr_name>의 layer_idxs 블록들에 LoRA 어댑터를 주입한다.

    peft의 layers_pattern 매칭은 named_modules() 키에서 '.<pattern>.<idx>.'를 정규식으로 찾으므로,
    scope_module은 블록 리스트를 담은 속성의 부모여야 한다 (블록 리스트를 가진 모듈 자신을
    스코프로 주면 키가 '<block_attr_name>.<idx>...'로 시작해 앞에 점(.)이 없어 매칭에 실패한다).
    """
    lora_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules,
        layers_to_transform=layer_idxs,
        layers_pattern=block_attr_name,
        bias='none',
    )
    inject_adapter_in_model(lora_cfg, scope_module, adapter_name='default')


def _mark_lora_trainable(scope_module):
    """LoRA 파라미터를 학습 대상으로 표시한다.

    peft의 inject_adapter_in_model은 내부적으로 BaseTuner._mark_only_adapters_as_trainable을
    호출해 scope_module 안의 '어댑터가 아닌 모든 파라미터'의 requires_grad를 무조건 False로
    되돌려버린다 (LoRA 대상 블록뿐 아니라 unfreeze로 켜둔 다른 블록/norm 등 scope 전체 포함).
    그래서 is_cnn_param=True(unfreeze로 학습 대상 표시된) 파라미터는 여기서 다시 True로
    복원해야 한다 — 단순히 "건드리지 않는" 것만으로는 이미 꺼진 값을 되살릴 수 없다.
    """
    for name, p in scope_module.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            p.requires_grad = True
            p.is_lora_param = True
        elif getattr(p, 'is_cnn_param', False):
            p.requires_grad = True
        else:
            p.requires_grad = False


class BiomedCLIPMultiModal(nn.Module):
    """이미지+텍스트를 BiomedCLIP(freeze)으로 인코딩 후 fusion -> classifier로 4-class 분류."""

    def __init__(self, mdlParams):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_TAG)
        self.clip = clip_model
        self.freeze = mdlParams.get('freeze_biomedclip', True)
        unfreeze_visual_blocks = mdlParams.get('unfreeze_visual_blocks', 0)
        unfreeze_text_blocks = mdlParams.get('unfreeze_text_blocks', 0)
        use_lora_visual = mdlParams.get('use_lora_visual', False)
        use_lora_text = mdlParams.get('use_lora_text', False)
        lora_visual_blocks = mdlParams.get('lora_visual_blocks', 0)
        lora_text_blocks = mdlParams.get('lora_text_blocks', 0)
        lora_r = mdlParams.get('lora_r', 8)
        lora_alpha = mdlParams.get('lora_alpha', 16)
        lora_dropout = mdlParams.get('lora_dropout', 0.05)

        if use_lora_visual and unfreeze_visual_blocks > 0:
            raise ValueError("use_lora_visual=True와 unfreeze_visual_blocks>0을 동시에 설정할 수 없습니다.")
        if use_lora_text and unfreeze_text_blocks > 0:
            raise ValueError("use_lora_text=True와 unfreeze_text_blocks>0을 동시에 설정할 수 없습니다.")

        self.visual_trainable = unfreeze_visual_blocks > 0 or use_lora_visual
        self.text_trainable = unfreeze_text_blocks > 0 or use_lora_text

        if self.freeze:
            for p in self.clip.parameters():
                p.requires_grad = False
            # 이미지 인코더(ViT trunk)의 마지막 N개 블록만 재활성화.
            if unfreeze_visual_blocks > 0:
                trunk = self.clip.visual.trunk
                unfrozen_modules = list(trunk.blocks[-unfreeze_visual_blocks:]) + [trunk.norm]
                for module in unfrozen_modules:
                    for p in module.parameters():
                        p.requires_grad = True
                        p.is_cnn_param = True
            # 텍스트 인코더(PubMedBERT)의 마지막 N개 block(layer)만 재활성화.
            if unfreeze_text_blocks > 0:
                bert_encoder = self.clip.text.transformer.encoder
                unfrozen_modules = list(bert_encoder.layer[-unfreeze_text_blocks:]) + [self.clip.text.proj]
                for module in unfrozen_modules:
                    for p in module.parameters():
                        p.requires_grad = True
                        p.is_cnn_param = True
            # 이미지 인코더(ViT trunk)의 LoRA 대상 블록에 어댑터 주입.
            # unfreeze_visual_blocks와 함께 켜져 있으면, unfreeze로 풀 파인튜닝되는 마지막
            # unfreeze_visual_blocks개 블록 바로 앞의 lora_visual_blocks개 블록에 적용된다
            # (범위가 겹치지 않도록 _lora_layer_range가 오프셋을 계산).
            if use_lora_visual:
                trunk = self.clip.visual.trunk
                layer_idxs = _lora_layer_range(len(trunk.blocks), unfreeze_visual_blocks, lora_visual_blocks)
                _inject_lora(self.clip.visual, 'blocks', layer_idxs,
                             ['qkv', 'proj', 'fc1', 'fc2'], lora_r, lora_alpha, lora_dropout)
                _mark_lora_trainable(self.clip.visual)
            # 텍스트 인코더(PubMedBERT)의 LoRA 대상 block(layer)에 어댑터 주입. 위와 동일하게
            # unfreeze_text_blocks와 겹치지 않는 앞쪽 구간을 사용.
            if use_lora_text:
                bert_encoder = self.clip.text.transformer.encoder
                layer_idxs = _lora_layer_range(len(bert_encoder.layer), unfreeze_text_blocks, lora_text_blocks)
                _inject_lora(self.clip.text.transformer, 'layer', layer_idxs,
                             ['query', 'key', 'value', 'dense'], lora_r, lora_alpha, lora_dropout)
                _mark_lora_trainable(self.clip.text.transformer)

        self.fusion_method = mdlParams.get('fusion_method', 'concat')
        # modality: 'multimodal'(기본) / 'image_only' / 'text_only'
        self.modality = mdlParams.get('modality', 'multimodal')
        if self.modality == 'multimodal':
            fusion_in = BIOMEDCLIP_EMBED_DIM * 2 if self.fusion_method == 'concat' else BIOMEDCLIP_EMBED_DIM
        else:
            fusion_in = BIOMEDCLIP_EMBED_DIM

        fc_after = mdlParams.get('fc_layers_after', [])
        if fc_after:
            self.meta_after = nn.Sequential(
                nn.Linear(fusion_in, fc_after[0]),
                nn.BatchNorm1d(fc_after[0]),
                nn.ReLU())
            classifier_in = fc_after[0]
        else:
            self.meta_after = None
            classifier_in = fusion_in
        self.classifier = nn.Linear(classifier_in, mdlParams['numClasses'])

    def forward(self, inputs):
        x, text_tokens = inputs
        text_tokens = text_tokens.long()

        if self.modality != 'text_only':
            if self.freeze and not self.visual_trainable:
                with torch.no_grad():
                    image_features = self.clip.encode_image(x, normalize=True)
            else:
                image_features = self.clip.encode_image(x, normalize=True)

        if self.modality != 'image_only':
            if self.freeze and not self.text_trainable:
                with torch.no_grad():
                    text_features = self.clip.encode_text(text_tokens, normalize=True)
            else:
                text_features = self.clip.encode_text(text_tokens, normalize=True)

        if self.modality == 'image_only':
            features = image_features
        elif self.modality == 'text_only':
            features = text_features
        elif self.fusion_method == 'elementwise':
            features = image_features * text_features
        elif self.fusion_method == 'average':
            features = (image_features + text_features) / 2
        else:
            features = torch.cat((image_features, text_features), dim=1)

        if self.meta_after is not None:
            features = self.meta_after(features)
        return self.classifier(features)