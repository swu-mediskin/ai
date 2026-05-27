import os
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
import shutil
from pathlib import Path


# 설정
IMG_DIR = "data/imgs"
CSV_PATH = "data/metadata.csv"
OUTPUT_IMG_DIR = "data/imgs_augmented"       
OUTPUT_CSV_PATH = "data/metadata_augmented.csv"  
TARGET_COUNT = 400                            
MAJORITY_CLASS = "BCC"                        
SEED = 42

np.random.seed(SEED)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

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

# 원본 데이터 로드
df = pd.read_csv(CSV_PATH)
print(f"원본 데이터 분포:\n{df['diagnostic'].value_counts()}\n")

# 원본 이미지를 output 폴더에 복사
print("원본 이미지 복사 중...")
for img_id in df['img_id']:
    src = os.path.join(IMG_DIR, img_id)
    dst = os.path.join(OUTPUT_IMG_DIR, img_id)
    if os.path.exists(src):
        shutil.copy2(src, dst)

augmented_rows = []


# 소수 클래스별 증강
minority_classes = [c for c in df['diagnostic'].unique() if c != MAJORITY_CLASS]

for cls in minority_classes:
    cls_df = df[df['diagnostic'] == cls].reset_index(drop=True)
    current_count = len(cls_df)
    needed = TARGET_COUNT - current_count

    if needed <= 0:
        print(f"{cls}: 이미 {current_count}장 → 증강 불필요")
        continue

    print(f"{cls}: {current_count}장 → {TARGET_COUNT}장 목표 ({needed}장 생성)")

    generated = 0
    while generated < needed:
        # 원본에서 랜덤 샘플 선택
        row = cls_df.iloc[generated % current_count].copy()
        img_id = row['img_id']
        img_path = os.path.join(IMG_DIR, img_id)

        if not os.path.exists(img_path):
            print(f"  이미지 없음: {img_path}")
            generated += 1
            continue

        # 이미지 증강
        image = np.array(Image.open(img_path).convert("RGB"))
        augmented = transform(image=image)['image']
        aug_image = Image.fromarray(augmented)

        # 새 파일명 생성
        ext = Path(img_id).suffix
        new_img_id = f"{Path(img_id).stem}_aug{generated}{ext}"
        aug_save_path = os.path.join(OUTPUT_IMG_DIR, new_img_id)
        aug_image.save(aug_save_path)

        # 메타데이터는 원본 그대로 복사 + img_id만 변경
        new_row = row.copy()
        new_row['img_id'] = new_img_id
        augmented_rows.append(new_row)

        generated += 1

# 최종 CSV 저장
aug_df = pd.DataFrame(augmented_rows)
final_df = pd.concat([df, aug_df], ignore_index=True)
final_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n증강 후 데이터 분포:")
print(final_df['diagnostic'].value_counts())
print(f"\n저장 완료 → {OUTPUT_CSV_PATH}")
print(f"이미지 저장 완료 → {OUTPUT_IMG_DIR}")