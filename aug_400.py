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
OUTPUT_BASE_DIR = "data/folds"   # fold별로 저장할 기본 경로
TARGET_COUNT = 400
MAJORITY_CLASS = "BCC"
NUM_FOLDS = 5
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
# 원본 데이터 로드
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df[df['diagnostic'].isin(['NEV', 'BCC', 'SCC', 'MEL'])].reset_index(drop=True)
print(f"원본 데이터 분포:\n{df['diagnostic'].value_counts()}\n")

# fold 분리에 필요한 정보 추출
patient_ids = df['patient_id'].values
class_indices = np.array([diag_map[d] for d in df['diagnostic']])
img_indices = np.arange(len(df))

# ============================================================
# StratifiedGroupKFold로 fold 분리
# ============================================================
sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(img_indices, class_indices, groups=patient_ids)):
    print(f"\n{'='*50}")
    print(f"Fold {fold} 처리 중... (Train: {len(train_idx)}장 | Val: {len(val_idx)}장)")
    print(f"{'='*50}")

    # fold별 출력 경로 생성
    fold_img_dir = os.path.join(OUTPUT_BASE_DIR, f"fold{fold}", "imgs")
    os.makedirs(fold_img_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # --------------------------------------------------------
    # 원본 이미지 전체 복사 (train + val)
    # --------------------------------------------------------
    print("원본 이미지 복사 중...")
    for img_id in df['img_id']:
        src = os.path.join(IMG_DIR, img_id)
        dst = os.path.join(fold_img_dir, img_id)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # --------------------------------------------------------
    # Train에만 증강 적용
    # --------------------------------------------------------
    augmented_rows = []
    minority_classes = [c for c in train_df['diagnostic'].unique() if c != MAJORITY_CLASS]

    for cls in minority_classes:
        cls_df = train_df[train_df['diagnostic'] == cls].reset_index(drop=True)
        current_count = len(cls_df)
        needed = TARGET_COUNT - current_count

        if needed <= 0:
            print(f"  {cls}: 이미 {current_count}장 → 증강 불필요")
            continue

        print(f"  {cls}: {current_count}장 → {TARGET_COUNT}장 목표 ({needed}장 생성)")

        generated = 0
        while generated < needed:
            row = cls_df.iloc[generated % current_count].copy()
            img_id = row['img_id']
            img_path = os.path.join(IMG_DIR, img_id)

            if not os.path.exists(img_path):
                print(f"    이미지 없음: {img_path}")
                generated += 1
                continue

            # 이미지 증강
            image = np.array(Image.open(img_path).convert("RGB"))
            augmented = transform(image=image)['image']
            aug_image = Image.fromarray(augmented)

            # 새 파일명 생성
            ext = Path(img_id).suffix
            new_img_id = f"{Path(img_id).stem}_aug{generated}{ext}"
            aug_save_path = os.path.join(fold_img_dir, new_img_id)
            aug_image.save(aug_save_path)

            # 메타데이터 복사 + img_id만 변경
            new_row = row.copy()
            new_row['img_id'] = new_img_id
            augmented_rows.append(new_row)

            generated += 1

    # --------------------------------------------------------
    # fold별 CSV 저장 (train/val 구분)
    # --------------------------------------------------------
    aug_df = pd.DataFrame(augmented_rows)
    final_train_df = pd.concat([train_df, aug_df], ignore_index=True)

    fold_dir = os.path.join(OUTPUT_BASE_DIR, f"fold{fold}")
    final_train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

    print(f"\n  [Fold {fold}] Train 증강 후 분포:")
    print(final_train_df['diagnostic'].value_counts().to_string())
    print(f"\n  [Fold {fold}] Val 분포 (원본만):")
    print(val_df['diagnostic'].value_counts().to_string())
    print(f"\n  저장 완료 → {fold_dir}")

print(f"\n전체 {NUM_FOLDS}개 Fold 처리 완료!")
print(f"결과 구조: data/folds/fold{{0~{NUM_FOLDS-1}}}/imgs, train.csv, val.csv")