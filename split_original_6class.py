"""6클래스(NEV/BCC/SCC/MEL/ACK/SEK) 원본 데이터를 5-fold로 분할해 data/original/fold{k}/ 에 저장.

- 기존 4클래스(NEV/BCC/SCC/MEL)는 data/folds/fold{k}/val.csv 의 배정을 그대로 사용
  (지금까지의 4클래스 실험과 val 샘플이 동일하도록).
- 새 클래스(ACK/SEK)는 환자 단위로 배정:
  * 4클래스 데이터에도 있는 환자 → 그 환자가 이미 속한 fold
  * 나머지 환자 → StratifiedGroupKFold(환자 그룹, 클래스 층화)로 5개 fold에 분배
- 증강 없음(원본만). 이미지는 data/imgs 의 하드링크(디스크 추가 사용 없음, 일반 파일처럼 동작).

출력: data/original/fold{k}/{imgs/, train.csv, val.csv}
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SRC_IMG_DIR = os.path.join(DATA_DIR, 'imgs')
OLD_FOLD_DIR = os.path.join(DATA_DIR, 'folds')
OUT_DIR = os.path.join(DATA_DIR, 'original')
NUM_FOLDS = 5
SEED = 42
OLD_CLASSES = ['NEV', 'BCC', 'SCC', 'MEL']
NEW_CLASSES = ['ACK', 'SEK']

meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

# 1) 기존 4클래스 fold 배정 (val.csv 기준)
img2fold = {}
for k in range(NUM_FOLDS):
    for img_id in pd.read_csv(os.path.join(OLD_FOLD_DIR, f'fold{k}', 'val.csv'))['img_id']:
        img2fold[img_id] = k
old = meta[meta['diagnostic'].isin(OLD_CLASSES)]
assert set(old['img_id']) == set(img2fold), '기존 fold val.csv가 4클래스 전체를 덮지 않음'
pat2fold = {}
for pid, img_id in zip(old['patient_id'], old['img_id']):
    assert pat2fold.setdefault(pid, img2fold[img_id]) == img2fold[img_id], f'환자 {pid}가 여러 fold에 있음'

# 2) 새 클래스: 겹치는 환자는 기존 fold, 나머지는 StratifiedGroupKFold
new = meta[meta['diagnostic'].isin(NEW_CLASSES)].copy()
new['fold'] = new['patient_id'].map(pat2fold)
rest = new[new['fold'].isna()]
sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
for k, (_, val_idx) in enumerate(sgkf.split(rest, rest['diagnostic'], groups=rest['patient_id'])):
    new.loc[rest.index[val_idx], 'fold'] = k
new['fold'] = new['fold'].astype(int)
print(f"ACK/SEK 환자 {new['patient_id'].nunique()}명 중 기존 fold로 배정: "
      f"{new[new['patient_id'].isin(pat2fold)]['patient_id'].nunique()}명")

df = meta[meta['diagnostic'].isin(OLD_CLASSES + NEW_CLASSES)].copy()
df['fold'] = df['img_id'].map(img2fold)
df.loc[new.index, 'fold'] = new['fold']
assert df['fold'].notna().all()
df['fold'] = df['fold'].astype(int)
assert (df.groupby('patient_id')['fold'].nunique() == 1).all(), '환자 누설: 한 환자가 여러 fold에 있음'

# 3) fold별 저장
cols = list(meta.columns)
for k in range(NUM_FOLDS):
    fold_dir = os.path.join(OUT_DIR, f'fold{k}')
    img_dir = os.path.join(fold_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    for img_id in df['img_id']:
        dst = os.path.join(img_dir, img_id)
        if not os.path.exists(dst):
            os.link(os.path.join(SRC_IMG_DIR, img_id), dst)
    train_df = df[df['fold'] != k][cols].reset_index(drop=True)
    val_df = df[df['fold'] == k][cols].reset_index(drop=True)
    train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)

    # 기존 4클래스 val과 동일한지 확인
    old_val = set(pd.read_csv(os.path.join(OLD_FOLD_DIR, f'fold{k}', 'val.csv'))['img_id'])
    assert set(val_df[val_df['diagnostic'].isin(OLD_CLASSES)]['img_id']) == old_val
    print(f"[fold{k}] train {len(train_df)} {train_df['diagnostic'].value_counts().to_dict()}")
    print(f"        val   {len(val_df)} {val_df['diagnostic'].value_counts().to_dict()}")
