# Mediskin — 피부 병변 멀티모달 분류

임상 피부 사진과 환자 메타데이터(나이, 성별, 병변 부위, 증상 등)를 함께 사용해 피부 병변을 분류하는 졸업 프로젝트입니다.
학습된 모델은 FastAPI 서버(`main.py`)로 서빙되며, Docker 이미지로 배포됩니다.

- **데이터셋**: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) — 스마트폰으로 촬영한 임상 이미지 2,298장 + 환자/병변 메타데이터
- **분류 대상**: 기본 4클래스 `NEV / BCC / SCC / MEL` (6클래스 실험 시 `ACK / SEK` 추가)
- **주요 모델**: [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (ViT-B/16 + PubMedBERT) 기반 이미지·텍스트 멀티모달 분류기
- **실험 기록**: Weights & Biases 프로젝트 `mediskin`

> 이 저장소는 ISIC 2019 챌린지 1위 솔루션인 DAISYLab 코드([ngessert/isic2019](https://github.com/ngessert/isic2019))를 기반으로 시작했습니다. 원본 학습/평가 파이프라인 구조(`mdlParams` 설정 시스템, `eval.py`, `ensemble.py`)는 대부분 그대로 유지하고 있습니다. 자세한 내용은 [Acknowledgements](#acknowledgements)를 참고하세요.

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [환경 설정](#환경-설정)
3. [데이터 준비](#데이터-준비)
4. [모델](#모델)
5. [학습](#학습)
6. [평가 및 분석](#평가-및-분석)
7. [서빙 (FastAPI / Docker)](#서빙-fastapi--docker)
8. [실험 시 주의사항](#실험-시-주의사항)
9. [Acknowledgements](#acknowledgements)

---

## 프로젝트 구조

```
.
├── train.py                  # 학습 진입점 (CNN / 메타데이터 / BiomedCLIP 공통)
├── train_meta.py             # 메타데이터 브랜치 학습 (CNN 가중치 로드 후 fusion 학습)
├── sweep_train.py            # wandb sweep 실행용 래퍼
├── eval.py / ensemble.py     # 원본 DAISYLab 평가·앙상블 스크립트
├── models.py                 # 백본 정의, modify_meta, MetaOnlyModel, BiomedCLIPMultiModal
├── utils.py                  # PADDataset, 손실함수, 평가 지표(F1, ECE, Brier 등)
├── auto_augment.py           # AutoAugment 정책 (utils.py에서 사용)
│
├── cfgs/2019/                # 실험별 설정 파일
├── pc_cfgs/                  # 머신별 경로 설정 (example.py 참고)
├── sweep_biomedclip_fusion.yaml  # wandb sweep 설정
│
│   # 메타데이터 → 텍스트
├── meta2text.py              # 템플릿 문장 변환
├── meta2text_llm.py          # GPT로 진단서 형식 문장 생성 (data/*_cache.json에 캐시)
├── validate_llm_cache.py     # LLM 생성 문장 검증 (원본 메타데이터와 사실 불일치 체크)
│
│   # 데이터 분할·증강 (한 번 실행해서 data/ 아래 생성)
├── aug_400.py                # 5-fold 분할 + 클래스별 400장 목표 증강 → data/folds/
├── aug_max.py                # 최대 클래스 수에 맞춘 증강 → data/max/
├── aug_mel.py                # MEL 집중 증강 → data/mel/
├── split_original_6class.py  # 6클래스 원본(증강 없음) 5-fold 분할 → data/original/
├── split_data.py             # 초기 데이터 분할 스크립트
│
│   # 분석·시각화
├── analyze_meta.py           # 메타데이터 컬럼–클래스 연관도 (ANOVA / Cramér's V)
├── gradcam.py                # Grad-CAM 시각화 (CNN / BiomedCLIP ViT)
├── wandb_log.py              # wandb run에서 best epoch 지표 조회
├── token_test.py             # BiomedCLIP 토크나이저 확인용
│
│   # Mediskin API 서버
├── main.py                   # FastAPI 앱 (/predict)
├── inference.py              # API 입력 → 모델 입력 변환 (메타 벡터 / 텍스트 토큰)
├── Dockerfile, docker-compose.yml
│
└── indices_isic2019.pkl, meta_data/   # 원본 DAISYLab ISIC 2019 예제용 (test_effb0_* cfg)
```

> 모든 스크립트는 저장소 루트에 있으며, 루트에서 `python <스크립트>.py` 형식으로 실행합니다.


## 환경 설정

Python 3.10 기준입니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- `requirements.txt`는 CPU용 PyTorch 인덱스를 사용합니다. CUDA 환경이라면 PyTorch를 별도로 설치하세요.
- Apple Silicon에서는 디바이스 인자로 `mps`를 주면 MPS 백엔드를 사용합니다.
- 학습 로그는 wandb로 전송되므로 `wandb login`이 필요합니다.
- LLM 문장 생성(`meta2text_llm.py`)을 사용할 경우 `.env`에 `OPENAI_API_KEY`를 설정합니다. 모델은 `LLM_META_MODEL` 환경변수로 바꿀 수 있습니다.

### 머신 설정 (`pc_cfgs/`)

`pc_cfgs/example.py`를 복사해 자신의 경로를 지정합니다. (`pc_cfgs/pad_cfg.py`는 git에서 제외됨)

```python
# pc_cfgs/pad_cfg.py
mdlParams = {}
mdlParams['pathBase'] = '/path/to/isic2019'   # data/, results/ 가 있는 위치
```

## 데이터 준비

PAD-UFES-20을 내려받아 아래처럼 배치합니다. (`data/`는 git에서 제외됨)

```
data/
├── metadata.csv     # PAD-UFES-20 메타데이터 (patient_id, img_id, diagnostic, age, region, ...)
└── imgs/            # PAD-UFES-20 이미지 (PAT_xxxx_xxxx_xxx.png)
```

| 클래스 | 설명 | 이미지 수 |
|---|---|---|
| BCC | 기저세포암 | 845 |
| ACK | 광선각화증 | 730 |
| NEV | 모반 | 244 |
| SEK | 지루각화증 | 235 |
| SCC | 편평세포암 | 192 |
| MEL | 흑색종 | 52 |

### Fold 분할

모든 분할은 **환자 단위**(`StratifiedGroupKFold`, seed 42)로 이루어져 같은 환자의 이미지가 train/val에 동시에 들어가지 않습니다.

```bash
# 4클래스 + 클래스별 400장 목표 증강 → data/folds/fold{0-4}/{imgs, train.csv, val.csv}
python aug_400.py

# 6클래스 원본(증강 없음) → data/original/fold{0-4}/...
# 기존 4클래스 val 배정을 그대로 유지하고 ACK/SEK만 새로 배정
python split_original_6class.py
```

증강은 train에만 적용되며 val은 원본 이미지만 사용합니다.

## 모델

### 1. BiomedCLIP 멀티모달 (`model_type = 'biomedclip'`) — 현재 주력

`models.BiomedCLIPMultiModal`

```
이미지 ──► BiomedCLIP ViT-B/16 ───► image_features (512) ─┐
                                                          ├─► Fusion ─► FC(1024) ─► Classifier
메타데이터 ─► 문장 ─► PubMedBERT ─► text_features (512) ──┘
```

- **메타데이터 → 텍스트**: `meta2text.py`의 `row2text`가 17개 임상 변수를 영어 문장으로 변환합니다.
  ```
  A clinical skin image of a 45-year-old female with Fitzpatrick skin type 3 showing a lesion on the face.
  The lesion measures approximately 6mm by 5mm. The patient is a smoker, ...
  ```
  `meta2text_llm.py`를 쓰면 GPT로 진단서 스타일 문장을 생성할 수 있습니다 (결과는 `data/*_cache.json`에 캐시).
- **modality**: `multimodal` / `image_only` / `text_only`
- **fusion_method**: `concat`(기본) / `elementwise` / `average` / `cross_attention` / `gated`(GMU) / `film`(FiLM)
- **파인튜닝 방식** (기본은 인코더 전체 freeze)
  - `unfreeze_visual_blocks`, `unfreeze_text_blocks`: 마지막 N개 블록 풀 파인튜닝
  - `use_lora_visual` / `use_lora_text` + `lora_*_blocks`: 마지막 N개 블록에 LoRA 적용 (PEFT, `r=16, alpha=32`)
- **손실함수**: CE / Weighted CE (`use_weighted_ce`) / Focal Loss / 한방향 Cost-Sensitive CE (`use_cost_sensitive_ce`, 예: SCC→BCC 오분류에 가중)

### 2. CNN + 메타데이터 벡터 (원본 계열)

- `models.getModel()`로 EfficientNet, DenseNet, ResNet, SE-ResNeXt 등 백본 선택
- `models.modify_meta()`가 메타데이터 one-hot/수치 벡터용 FC 브랜치를 붙여 이미지 특징과 결합
- `models.MetaOnlyModel`: 메타데이터만 사용하는 MLP

### 설정 파일 네이밍 (`cfgs/2019/`)

| 접두사 | 내용 |
|---|---|
| `pad_bio_*` | BiomedCLIP 기반 실험 (`_imgonly`, `_textonly`, `_lora_vit6`, `_gmu`, `_film`, `_crossattn`, `_focal`, `_cost`, `_llm`, `_nomiss`, `_6cls`, `_foldN` 등) |
| `pad_multi_*` | CNN + 메타데이터 벡터 멀티모달 |
| `pad_meta_only_*` | 메타데이터 단독 모델 |
| `pad_img_aug*` | 이미지 단독 + 증강 전략별 |
| `test_effb0_*` | 원본 DAISYLab EfficientNet-B0 예제 |

새 실험은 가장 가까운 cfg를 복사해 `mdlName`과 바꿀 하이퍼파라미터만 수정하는 것을 권장합니다.

## 학습

```bash
python train.py <pc_cfg> <model_cfg> <device> [cv_subset]

# BiomedCLIP 이미지 단독, ViT 마지막 6블록 LoRA (fold0)
python train.py pad_cfg 2019.pad_bio_lora_vit6_imgonly mps

# GMU fusion 멀티모달
python train.py pad_cfg 2019.pad_bio_lora_vit6_gmu mps

# 6클래스 학습, 4클래스 기준으로 best 선택
python train.py pad_cfg 2019.pad_bio_lora_vit6_imgonly_6cls mps
```

- `<device>`: `mps` → Apple GPU, `gpu`/`cuda` → `cuda:0`, 그 외 CPU
- 5-fold CV는 fold별 cfg(`*_fold1` ~ `*_fold4`)를 각각 실행합니다.
- **Early stopping / best checkpoint**는 validation **macro F1** 기준입니다. 6클래스 학습 시 `eval_classes`를 지정하면 해당 클래스(NEV/BCC/SCC/MEL)만으로 F1을 계산해 기존 4클래스 실험과 비교 가능하게 합니다.
- 체크포인트는 `results/<model_cfg>/<mdlName>/best_fold{k}.pt`에 저장됩니다.
- epoch마다 wandb에 accuracy, 클래스별 precision/recall/F1/AUC/AUPRC, **ECE, Brier score**, confusion matrix, PR/ROC curve를 기록합니다.

wandb sweep:

```bash
wandb sweep sweep_biomedclip_fusion.yaml
wandb agent <sweep_id>
```

## 평가 및 분석

```bash
# wandb run의 best epoch 지표 + 클래스별 지표 출력 (run id는 스크립트 안에서 수정)
python wandb_log.py

# Grad-CAM 시각화
python gradcam.py <pc_cfg> <model_cfg> <device> [best|last] [cv] [val|이미지경로] [n]
python gradcam.py pad_cfg 2019.pad_bio_lora_vit6_imgonly mps best 0 val 8

# 메타데이터 컬럼별 클래스 연관도 분석
python analyze_meta.py

# LLM 생성 문장 검증
python validate_llm_cache.py --cache data/new_llm_text_cache2.json --fold-dir data/folds/fold0
```

원본 DAISYLab의 multi-crop 평가(`eval.py`)와 앙상블(`ensemble.py`) 사용법은 [원본 저장소 README](https://github.com/ngessert/isic2019#readme)를 참고하세요.

## 서빙 (FastAPI / Docker)

### 로컬 실행

```bash
python main.py                          # 0.0.0.0:8000
uvicorn main:app --host 0.0.0.0 --port 8080
```

서빙할 모델은 `main.py` 상단 상수로 지정합니다.

```python
PC_CFG = "pad_cfg"
MODEL_CFG = "2019.pad_multi_all_real_aug"
DEVICE_STR = "mps"
```

체크포인트 경로는 `MODEL_WEIGHTS_PATH` 환경변수로 덮어쓸 수 있습니다.

### API

`POST /predict` — `multipart/form-data`

| 필드 | 타입 | 설명 |
|---|---|---|
| `file` | 이미지 | 병변 사진 |
| `age`, `fitspatrick`, `diameter_1`, `diameter_2` | float | 나이, Fitzpatrick 피부타입, 병변 크기(mm) |
| `gender`, `region` | str | 성별, 병변 부위 |
| `itch`, `grew`, `hurt`, `changed`, `bleed`, `elevation` | str | 증상 (`true`/`false`/`unk`, 대소문자 무관) |
| `smoke`, `drink`, `pesticide`, `skin_cancer_history`, `cancer_history` | str | 생활습관·병력 (`true`/`false`/`unk`, 대소문자 무관) |

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@lesion.png" -F "age=63" -F "gender=FEMALE" -F "region=FACE" \
  -F "itch=true" -F "grew=true" -F "bleed=false"
```

```json
{
  "prediction": "BCC",
  "confidence": 0.87,
  "all_probabilities": [0.03, 0.87, 0.08, 0.02]
}
```

`all_probabilities`의 순서는 `[NEV, BCC, SCC, MEL]`입니다.

### Docker

```bash
docker compose up
```

- `./results`, `./data`를 볼륨으로 마운트하고 `MODEL_WEIGHTS_PATH`, `TRAIN_CSV_PATH`, `DATA_PATH`를 설정합니다.
- 이미지에는 `main.py`, `inference.py`, `models.py`, `utils.py`, `cfgs/`, `pc_cfgs/`만 복사됩니다. 서빙 경로에서 이 밖의 파일을 import한다면 `Dockerfile`에도 추가해야 합니다.

## 실험 시 주의사항

- **메타데이터 결측 누설**: PAD-UFES-20에서 성별·피부타입·크기·병력 등이 **모두 결측인 샘플은 전부 비생검 NEV**입니다. 템플릿 문장은 결측 필드를 생략하므로 "문장이 짧다"는 것 자체가 NEV 신호가 되어 텍스트/멀티모달 모델 성능이 부풀려집니다. 이미지 단독과 비교할 때는 결측 NEV를 제외한 수치도 함께 보고하고, `*_nomiss` cfg(결측 샘플 제외 학습)를 참고하세요.
- **`biopsed` 피처 제외**: 진단 후에야 알 수 있는 값이라 데이터 누수이므로 사용하지 않습니다. `background_father/mother`, `has_piped_water`, `has_sewage_system`, `pesticide`처럼 실서비스에서 수집하기 어려운 피처도 `analyze_meta.py` 분석 후 제외한 설정(`*_analyzed_05`)이 있습니다.
- **학습–서빙 인코딩 동기화**: 학습 시 메타데이터 인코딩(벡터 차원, 문장 템플릿)을 바꾸면 `inference.py`의 `build_meta_vector` / `build_text_tokens`도 같이 수정해야 합니다.
- **클래스 수**: 서버는 4클래스(`NEV/BCC/SCC/MEL`)를 가정합니다. 6클래스 체크포인트를 서빙하려면 `main.py`의 `class_names`를 수정해야 합니다.
- **재현성**: 같은 설정을 다시 돌려도 fold0 macro F1이 ±0.04 정도 흔들립니다. 결론은 5-fold 평균으로 내는 것을 권장합니다.

## Acknowledgements

이 프로젝트는 아래 연구의 공개 코드를 기반으로 합니다.

- N. Gessert, M. Nielsen, M. Shaikh, R. Werner, A. Schlaefer, *Skin Lesion Classification Using Ensembles of Multi-Resolution EfficientNets with Meta Data*, MethodsX, 2020. [doi:10.1016/j.mex.2020.100864](https://doi.org/10.1016/j.mex.2020.100864) · [arXiv:1910.03910](https://arxiv.org/abs/1910.03910) · [GitHub](https://github.com/ngessert/isic2019)
- A. G. C. Pacheco et al., *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*, Data in Brief, 2020.
- S. Zhang et al., *BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs*, 2023.

라이선스는 [LICENSE](LICENSE)를 따릅니다.
