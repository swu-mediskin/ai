import os
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
import shutil
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# 설정
# ============================================================
IMG_DIR = "data/imgs"
CSV_PATH = "data/metadata.csv"
OUTPUT_BASE_DIR = "data/mel"
TARGET_COUNT = 191  # MEL 목표 장수
SEED = 42

np.random.seed(SEED)

diag_map = {'NEV': 0, 'BCC': 1, 'SCC': 2, 'MEL': 3}

# ============================================================
# Augmentation 파이프라인
# ============================================================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.3),
    A.Resize(224, 224),
])

# ============================================================
# 원본 데이터 로드 및 필터링
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df[df['diagnostic'].isin(['NEV', 'BCC', 'SCC', 'MEL'])].reset_index(drop=True)
print(f"원본 데이터 분포:\n{df['diagnostic'].value_counts()}\n")

# fold 분리
patient_ids = df['patient_id'].values
class_indices = np.array([diag_map[d] for d in df['diagnostic']])
img_indices = np.arange(len(df))

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(sgkf.split(img_indices, class_indices, groups=patient_ids))

# fold0만 사용
train_idx, val_idx = splits[0]
fold = 0

print(f"Fold {fold} 처리 중... (Train: {len(train_idx)}장 | Val: {len(val_idx)}장)")

fold_img_dir = os.path.join(OUTPUT_BASE_DIR, f"fold{fold}", "imgs")
os.makedirs(fold_img_dir, exist_ok=True)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

# ============================================================
# 원본 이미지 전체 복사 (train + val)
# ============================================================
print("원본 이미지 복사 중...")
for img_id in df['img_id']:
    src = os.path.join(IMG_DIR, img_id)
    dst = os.path.join(fold_img_dir, img_id)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

# ============================================================
# MEL만 증강 적용
# ============================================================
augmented_rows = []

cls_df = train_df[train_df['diagnostic'] == 'MEL'].reset_index(drop=True)
current_count = len(cls_df)
needed = TARGET_COUNT - current_count

print(f"\nMEL: {current_count}장 → {TARGET_COUNT}장 목표 ({needed}장 생성)")

# 균등 샘플링
repeat_times = (needed // current_count) + 1
shuffled_indices = []
for _ in range(repeat_times):
    idx = np.arange(current_count)
    np.random.shuffle(idx)
    shuffled_indices.extend(idx.tolist())
shuffled_indices = shuffled_indices[:needed]

for generated, sample_idx in enumerate(shuffled_indices):
    row = cls_df.iloc[sample_idx].copy()
    img_id = row['img_id']
    img_path = os.path.join(IMG_DIR, img_id)

    if not os.path.exists(img_path):
        print(f"  이미지 없음: {img_path}")
        continue

    image = np.array(Image.open(img_path).convert("RGB"))
    augmented = transform(image=image)['image']
    aug_image = Image.fromarray(augmented)

    ext = Path(img_id).suffix
    new_img_id = f"{Path(img_id).stem}_aug{generated}{ext}"
    aug_image.save(os.path.join(fold_img_dir, new_img_id))

    new_row = row.copy()
    new_row['img_id'] = new_img_id
    augmented_rows.append(new_row)

# ============================================================
# CSV 저장
# ============================================================
aug_df = pd.DataFrame(augmented_rows)
final_train_df = pd.concat([train_df, aug_df], ignore_index=True)

fold_dir = os.path.join(OUTPUT_BASE_DIR, f"fold{fold}")
final_train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

print(f"\nTrain 증강 후 분포:")
print(final_train_df['diagnostic'].value_counts())
print(f"\nVal 분포 (원본만):")
print(val_df['diagnostic'].value_counts())
print(f"\n저장 완료 → {fold_dir}")