import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'metadata.csv')

CLASSES = ['NEV', 'BCC', 'SCC', 'MEL']

df = pd.read_csv(CSV_PATH)
df = df[df['diagnostic'].isin(CLASSES)].reset_index(drop=True)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(df, df['diagnostic'], groups=df['patient_id']))
train_idx, val_idx = splits[0]

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

assert set(train_df['patient_id']) & set(val_df['patient_id']) == set()

train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(DATA_DIR, 'val.csv'), index=False)

print(f"Train: {len(train_df)} rows, {train_df['patient_id'].nunique()} patients")
print(train_df['diagnostic'].value_counts())
print(f"\nVal:   {len(val_df)} rows, {val_df['patient_id'].nunique()} patients")
print(val_df['diagnostic'].value_counts())
