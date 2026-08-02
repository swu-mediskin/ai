import os
import numpy as np
import pandas as pd

"""
기존 연구(pad_meta_all_real.py)가 쓰던 17개 임상 변수를 전부 반영한다.
수치형 age, fitspatrick, diameter_1, diameter_2,
범주형 2 gender,
범주형 14 region,
범주형 2 smoke, drink, pesticide, skin_cancer_history, cancer_history, itch, grew, hurt, changed, bleed, elevation

전체 문장
A clinical skin image of a 45-year-old female with Fitzpatrick skin type 3 showing a lesion on the face. The lesion measures approximately 6mm by 5mm. The patient is a smoker, drinks alcohol, has been exposed to pesticides, has a history of skin cancer, and has a personal history of cancer. The lesion has been itching, growing, hurting, changing, bleeding, elevating.
"""

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'metadata.csv')

SYMPTOM_LABELS = {
    'itch': 'itching',
    'grew': 'growing',
    'hurt': 'hurting',
    'changed': 'changing',
    'bleed': 'bleeding',
    'elevation': 'elevating',
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


def _join_natural(items):
    """['a'] -> 'a', ['a','b'] -> 'a and b', ['a','b','c'] -> 'a, b, and c'."""
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def row2text(row):
    """metadata.csv 한 행을 자연어 문장으로 변환."""
    parts = []

    age = row.get('age')
    gender_val = row.get('gender')
    gender_word = {'MALE': 'male', 'FEMALE': 'female'}.get(gender_val, 'patient')
    fitspatrick = row.get('fitspatrick')

    intro = "A clinical skin image of a"
    intro += f" {int(age)}-year-old" if pd.notna(age) else ""
    intro += f" {gender_word}"
    if pd.notna(fitspatrick):
        intro += f" with Fitzpatrick skin type {int(fitspatrick)}"
    parts.append(intro)

    region = row.get('region')
    if pd.notna(region):
        parts.append(f"showing a lesion on the {str(region).lower()}.")
    else:
        parts.append("showing a lesion.")

    diameter_1 = row.get('diameter_1')
    diameter_2 = row.get('diameter_2')
    if pd.notna(diameter_1) and pd.notna(diameter_2):
        parts.append(f"The lesion measures approximately {diameter_1:g}mm by {diameter_2:g}mm.")

    history_bits = []
    smoker = _to_bool(row.get('smoke'))
    if smoker is True:
        history_bits.append("is a smoker")
    elif smoker is False:
        history_bits.append("is not a smoker")

    drinker = _to_bool(row.get('drink'))
    if drinker is True:
        history_bits.append("drinks alcohol")
    elif drinker is False:
        history_bits.append("does not drink alcohol")

    pesticide_exp = _to_bool(row.get('pesticide'))
    if pesticide_exp is True:
        history_bits.append("has been exposed to pesticides")
    elif pesticide_exp is False:
        history_bits.append("has not been exposed to pesticides")

    cancer_hist = _to_bool(row.get('skin_cancer_history'))
    if cancer_hist is True:
        history_bits.append("has a history of skin cancer")
    elif cancer_hist is False:
        history_bits.append("has no history of skin cancer")

    general_cancer_hist = _to_bool(row.get('cancer_history'))
    if general_cancer_hist is True:
        history_bits.append("has a personal history of cancer")
    elif general_cancer_hist is False:
        history_bits.append("has no personal history of cancer")

    if history_bits:
        parts.append("The patient " + _join_natural(history_bits) + ".")

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


def row2meta_dict(row):
    """metadata.csv 한 행에서 GPT에 넘길 raw 메타데이터 dict를 만든다.
    값이 없는(NaN/UNK) 필드는 생략한다."""
    meta = {}

    age = row.get('age')
    if pd.notna(age):
        meta['age'] = int(age)

    gender = row.get('gender')
    if pd.notna(gender):
        meta['gender'] = str(gender).lower()

    region = row.get('region')
    if pd.notna(region):
        meta['region'] = str(region).lower()

    fitspatrick = row.get('fitspatrick')
    if pd.notna(fitspatrick):
        meta['fitzpatrick_skin_type'] = int(fitspatrick)

    diameter_1 = row.get('diameter_1')
    if pd.notna(diameter_1):
        meta['diameter_1_mm'] = float(diameter_1)

    diameter_2 = row.get('diameter_2')
    if pd.notna(diameter_2):
        meta['diameter_2_mm'] = float(diameter_2)

    smoker = _to_bool(row.get('smoke'))
    if smoker is not None:
        meta['smoker'] = smoker

    drinker = _to_bool(row.get('drink'))
    if drinker is not None:
        meta['drinks_alcohol'] = drinker

    pesticide_exp = _to_bool(row.get('pesticide'))
    if pesticide_exp is not None:
        meta['pesticide_exposure'] = pesticide_exp

    cancer_hist = _to_bool(row.get('skin_cancer_history'))
    if cancer_hist is not None:
        meta['skin_cancer_history'] = cancer_hist

    general_cancer_hist = _to_bool(row.get('cancer_history'))
    if general_cancer_hist is not None:
        meta['cancer_history'] = general_cancer_hist

    for col, label in SYMPTOM_LABELS.items():
        status = _to_bool(row.get(col))
        if status is not None:
            meta[label] = status

    return meta


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
