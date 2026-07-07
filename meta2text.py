import os
import numpy as np
import pandas as pd

"""
수치형 age, 
범주형 2 gender, 
범주형 14 region, 
범주형 2 smoke, skin_cancer_history, itch, grew, hurt, changed, bleed, elevation

A 
"""

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'metadata.csv')

SYMPTOM_LABELS = {
    'itch': 'itching',
    'grew': 'growing in size',
    'hurt': 'hurting',
    'changed': 'changing in appearance',
    'bleed': 'bleeding',
    'elevation': 'raised (elevated)',
}


def _to_bool(value):
    """smoke/skin_cancer_history(NaN/True/False)와 itch/grew/...('TRUE'/'FALSE'/'UNK')를
    공통된 True/False/None으로 정규화."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        if value == 'UNK':
            return None
        return value == 'TRUE'
    return bool(value)


def row2text(row):
    """metadata.csv 한 행을 자연어 문장으로 변환."""
    parts = []

    age = row.get('age')
    gender_val = row.get('gender')
    gender_word = {'MALE': 'male', 'FEMALE': 'female'}.get(gender_val, 'patient')

    if pd.notna(age):
        parts.append(f"A {int(age)}-year-old {gender_word}")
    else:
        parts.append(f"A {gender_word}")

    region = row.get('region')
    if pd.notna(region):
        parts.append(f"presents with a lesion on the {str(region).lower()}.")
    else:
        parts.append("presents with a lesion.")

    history_bits = []
    smoker = _to_bool(row.get('smoke'))
    if smoker is True:
        history_bits.append("is a smoker")
    elif smoker is False:
        history_bits.append("is not a smoker")

    cancer_hist = _to_bool(row.get('skin_cancer_history'))
    if cancer_hist is True:
        history_bits.append("has a history of skin cancer")
    elif cancer_hist is False:
        history_bits.append("has no history of skin cancer")

    if history_bits:
        parts.append("The patient " + " and ".join(history_bits) + ".")

    present_symptoms = []
    absent_symptoms = []
    for col, label in SYMPTOM_LABELS.items():
        status = _to_bool(row.get(col))
        if status is True:
            present_symptoms.append(label)
        elif status is False:
            absent_symptoms.append(label)

    if present_symptoms:
        parts.append("The lesion has been " + ", ".join(present_symptoms) + ".")
    if absent_symptoms:
        parts.append("It has not been " + ", ".join(absent_symptoms) + ".")

    return " ".join(parts)


def meta2text():
    """
        BiomedCLIP에 text 인풋을 입력하기 위해 기존 메타데이터를 자연어로 변환하는 함수
    """
    df = pd.read_csv(CSV_PATH)
    df['text'] = df.apply(row2text, axis=1)

    for text in df['text'].head(5):
        print(text)

    return df


if __name__ == "__main__":
    meta2text()
