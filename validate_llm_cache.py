import re
import json
import argparse

import pandas as pd

from meta2text import SYMPTOM_LABELS, row2meta_dict
import meta2text_llm as mllm


FITZPATRICK_ROMAN = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}

GENDER_WORDS = {
    'male': [r'\bmale\b', r'\bman\b', r'\bboy\b', r'\bhe\b', r'\bhis\b', r'\bhim\b'],
    'female': [r'\bfemale\b', r'\bwoman\b', r'\bgirl\b', r'\bshe\b', r'\bher\b'],
}
ANY_GENDER_WORDS = GENDER_WORDS['male'] + GENDER_WORDS['female']


ALL_REGIONS = ['face', 'back', 'chest', 'nose', 'forearm', 'arm', 'neck', 'ear',
               'hand', 'thigh', 'abdomen', 'lip', 'foot', 'scalp']
BOOL_FIELD_PATTERNS = {
    'smoker': r'smok(e|er|ing)',
    'drinks_alcohol': r'alcohol',
    'pesticide_exposure': r'pesticide',
    'skin_cancer_history': r'skin cancer',
    'cancer_history': r'\bcancer\b|malignan(t|cy)',
}

NEGATION_MARKERS = re.compile(r'\b(not|no|non-?|without|den(y|ies|ied)|absence of|negative for)\b', re.IGNORECASE)
NEGATION_WINDOW = 80  # 매칭된 단어 앞쪽 몇 글자까지 부정어를 찾을지
SENTENCE_BOUNDARY = re.compile(r'[.;]\s*')


def _is_negated_nearby(text, match_start):
    window_start = max(0, match_start - NEGATION_WINDOW)
    window = text[window_start:match_start]
    boundaries = list(SENTENCE_BOUNDARY.finditer(window))
    if boundaries:
        window = window[boundaries[-1].end():]
    return bool(NEGATION_MARKERS.search(window))

SYMPTOM_KEYWORDS = {
    'itch': r'itch|pruritus|pruritic',
    'grew': r'grow|grew|growth|enlarg',
    'hurt': r'hurt|pain|tender',
    'changed': r'chang(e|ing|ed)|evolv',
    'bleed': r'bleed|bleeding|hemorrhag',
    'elevation': r'elevat|raised|\brais(e|ing|ed)\b',
}

DIAGNOSIS_LEAKAGE_TERMS = [
    r'melanoma', r'\bnevus\b', r'\bnaevus\b', r'basal cell carcinoma', r'\bbcc\b',
    r'squamous cell carcinoma', r'\bscc\b', r'actinic keratosis', r'seborrheic keratosis',
]

TEMPLATE_FALLBACK_PREFIX = "A clinical skin image of a"


def _search_any(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check_entry(row, text):
    """(경고 목록, 체크한 필드 수, 커버된 필드 수, 환각 필드 수) 반환.

    "체크한/커버된" 필드 수는 메타데이터에 실제 값이 있는 필드만 대상으로 한다(coverage 지표용).
    환각 카운트는 그와 별개로, 메타데이터에 없는(NaN) 필드가 문장에 등장했는지를 전 필드에 대해 센다.
    """
    warnings = []
    meta = row2meta_dict(row)
    checked = 0
    covered = 0
    hallucinated = 0

    # 0. 템플릿 폴백 탐지
    if text.strip().startswith(TEMPLATE_FALLBACK_PREFIX):
        warnings.append("TEMPLATE_FALLBACK: GPT 호출 실패 시의 row2text() 폴백 문장으로 보임")

    # 0.5 진단명 누출 탐지
    leaked = [t for t in DIAGNOSIS_LEAKAGE_TERMS if re.search(t, text, re.IGNORECASE)]
    if leaked:
        warnings.append(f"DIAGNOSIS_LEAKAGE: 진단 관련 단어 발견 {leaked}")

    # 1. age
    if 'age' in meta:
        checked += 1
        if re.search(rf'\b{meta["age"]}\b', text):
            covered += 1
        else:
            warnings.append(f"MISSING: age={meta['age']}가 문장에 없음")
    else:
        if re.search(r'\b\d{1,3}[\s-]*(year|yr)s?[\s-]*old\b', text, re.IGNORECASE):
            hallucinated += 1
            warnings.append("HALLUCINATION: age가 메타데이터엔 없는데 문장엔 나이 표현이 있음")

    # 2. gender
    if 'gender' in meta:
        checked += 1
        words = GENDER_WORDS.get(meta['gender'], [])
        if _search_any(words, text):
            covered += 1
        else:
            warnings.append(f"MISSING: gender={meta['gender']}가 문장에 없음")
    else:
        if _search_any(ANY_GENDER_WORDS, text):
            hallucinated += 1
            warnings.append("HALLUCINATION: gender가 메타데이터엔 없는데 문장엔 성별 표현이 있음")

    # 3. region
    if 'region' in meta:
        checked += 1
        if re.search(rf'\b{re.escape(meta["region"])}\b', text, re.IGNORECASE):
            covered += 1
        else:
            warnings.append(f"MISSING: region={meta['region']}가 문장에 없음")
    else:
        hit = [r for r in ALL_REGIONS if re.search(rf'\b{r}\b', text, re.IGNORECASE)]
        if hit:
            hallucinated += 1
            warnings.append(f"HALLUCINATION: region이 메타데이터엔 없는데 문장엔 부위({hit}) 언급이 있음")

    # 4. fitzpatrick_skin_type (아라비아 숫자 또는 로마 숫자)
    if 'fitzpatrick_skin_type' in meta:
        checked += 1
        v = meta['fitzpatrick_skin_type']
        roman = FITZPATRICK_ROMAN.get(v, '')
        if re.search(rf'\b{v}\b', text) or (roman and re.search(rf'\b{roman}\b', text)):
            covered += 1
        else:
            warnings.append(f"MISSING: fitzpatrick_skin_type={v}(={roman})가 문장에 없음")
    else:
        if re.search(r'fitzpatrick', text, re.IGNORECASE):
            hallucinated += 1
            warnings.append("HALLUCINATION: fitzpatrick_skin_type이 메타데이터엔 없는데 문장엔 Fitzpatrick 언급이 있음")

    # 5. diameter_1_mm / diameter_2_mm (둘 중 하나라도 있으면 커버리지 체크, 둘 다 없으면 환각 체크)
    for key in ['diameter_1_mm', 'diameter_2_mm']:
        if key in meta:
            checked += 1
            v = meta[key]
            iv = int(v) if float(v) == int(v) else None
            patterns = [rf'\b{iv}\b', rf'\b{iv}\.0\b'] if iv is not None else [rf'\b{v:g}\b']
            if _search_any(patterns, text):
                covered += 1
            else:
                warnings.append(f"MISSING: {key}={v}가 문장에 없음")
    if 'diameter_1_mm' not in meta and 'diameter_2_mm' not in meta:
        if re.search(r'\bmm\b', text, re.IGNORECASE):
            hallucinated += 1
            warnings.append("HALLUCINATION: diameter가 메타데이터엔 없는데 문장엔 mm 단위 크기 언급이 있음")

    # 6. boolean 필드 (smoker, drinks_alcohol, pesticide_exposure, skin_cancer_history, cancer_history)
    for key, pattern in BOOL_FIELD_PATTERNS.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if key == 'cancer_history':
            # "skin cancer"에 포함된 'cancer'는 skin_cancer_history 소관이니 제외
            matches = [m for m in matches
                       if not re.search(r'skin\s*$', text[max(0, m.start() - 6):m.start()], re.IGNORECASE)]
        if key in meta:
            checked += 1
            value = meta[key]
            if not matches:
                warnings.append(f"MISSING: {key}={value}가 문장에 아예 언급 안 됨")
                continue
            covered += 1
            is_negated = _is_negated_nearby(text, matches[-1].start())
            expected_negated = (value is False)
            if is_negated != expected_negated:
                warnings.append(f"POLARITY_MAYBE_WRONG: {key}={value}인데 문장 뉘앙스가 반대로 읽힘(휴리스틱, 오탐 가능)")
        else:
            if matches:
                hallucinated += 1
                warnings.append(f"HALLUCINATION: {key}가 메타데이터엔 없는데 문장에 관련 언급이 있음")

    # 7. 증상 6개
    for col, keyword in SYMPTOM_KEYWORDS.items():
        is_absent = col not in row or pd.isna(row.get(col)) or row.get(col) == 'UNK'
        if is_absent:
            if re.search(keyword, text, re.IGNORECASE):
                hallucinated += 1
                warnings.append(f"HALLUCINATION: 증상 '{SYMPTOM_LABELS[col]}'이 메타데이터엔 없는데 문장에 언급됨")
            continue
        checked += 1
        if re.search(keyword, text, re.IGNORECASE):
            covered += 1
        else:
            warnings.append(f"MISSING: 증상 '{SYMPTOM_LABELS[col]}'이 문장에 없음 (원본값={row.get(col)})")

    return warnings, checked, covered, hallucinated


def _dedupe_cache_items(cache):
    """cache(dict)를 '_aug\\d+' 접미사 제거한 원본 이미지 기준으로 묶어서, 그룹당 대표 1개
    (원본이 있으면 원본, 없으면 그룹에서 캐시에 먼저 등장한 것)만 남긴 (img_id, text) 리스트를 반환.
    cache가 삽입 순서를 유지하는 걸 이용해 원래 순서를 최대한 보존한다.
    """
    groups = {}
    for img_id, text in cache.items():
        base_img_id = mllm._AUG_SUFFIX_RE.sub(r'\1', img_id)
        groups.setdefault(base_img_id, []).append((img_id, text))

    deduped = []
    for base_img_id, items in groups.items():
        rep = next((it for it in items if it[0] == base_img_id), items[0])
        deduped.append(rep)
    return deduped


def validate_cache(cache_path, df, max_print=20, limit=None, dedupe=True):
    with open(cache_path) as f:
        cache = json.load(f)

    items = _dedupe_cache_items(cache) if dedupe else list(cache.items())
    n_before_limit = len(items)
    if limit is not None:
        items = items[:limit]

    df_by_img = df.set_index('img_id')

    total_checked = 0
    total_covered = 0
    total_hallucinated = 0
    field_miss_count = {}
    flagged = []
    per_entry_ratios = []  # 문장(entry) 단위 커버리지 비율들
    perfect_entries = 0    # 빠진 필드가 0개인 문장 수
    n_valid_entries = 0    # df에서 찾은(검증 가능한) 문장 수

    for img_id, text in items:
        if img_id not in df_by_img.index:
            flagged.append((img_id, [f"NOT_IN_DF: {img_id}가 주어진 df에 없어서 검증 불가"]))
            continue
        row = df_by_img.loc[img_id]
        if isinstance(row, pd.DataFrame):  # 중복 img_id 방어
            row = row.iloc[0]
        warnings, checked, covered, hallucinated = check_entry(row, text)
        total_checked += checked
        total_covered += covered
        total_hallucinated += hallucinated
        n_valid_entries += 1
        if checked > 0:
            per_entry_ratios.append(covered / checked)
            if covered == checked:
                perfect_entries += 1
        for w in warnings:
            key = w.split(':')[0]
            field_miss_count[key] = field_miss_count.get(key, 0) + 1
        if warnings:
            flagged.append((img_id, warnings))

    macro_avg = sum(per_entry_ratios) / len(per_entry_ratios) if per_entry_ratios else 0.0

    print(f"=== {cache_path} 검증 결과 ===")
    print(f"전체 캐시 항목: {len(cache)}개"
          + (f" (aug 중복 제거 후 {n_before_limit}개" if dedupe else "")
          + (f", 상위 {len(items)}개만 검증)" if limit is not None else (")" if dedupe else "")))
    print(f"[필드 단위] 커버리지(micro-average): {total_covered}/{total_checked} ({100*total_covered/max(total_checked,1):.1f}%)")
    print(f"[문장 단위] 평균 커버리지(macro-average, 문장별 비율의 평균): {100*macro_avg:.1f}%")
    print(f"[문장 단위] 빠진 필드 0개(완벽) 문장: {perfect_entries}/{n_valid_entries} ({100*perfect_entries/max(n_valid_entries,1):.1f}%)")
    print(f"[환각] 메타데이터에 없는데 문장에 등장한 필드: 총 {total_hallucinated}건")
    print(f"경고 있는 항목: {len(flagged)}개")
    print(f"경고 유형별 건수: {field_miss_count}")
    print()
    print(f"--- 상위 {max_print}건 상세 (전체 {len(flagged)}건) ---")
    for img_id, warnings in flagged[:max_print]:
        print(f"[{img_id}]")
        for w in warnings:
            print(f"    {w}")

    return flagged


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default=mllm.CACHE_PATH)
    parser.add_argument('--fold-dir', default='data/folds/fold0')
    parser.add_argument('--max-print', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None, help='dedupe 후 상위 N개만 검증 (예: --limit 200)')
    parser.add_argument('--no-dedupe', action='store_true', help='aug 중복 제거 안 하고 캐시 원본 그대로 검증')
    args = parser.parse_args()

    train_df = pd.read_csv(f'{args.fold_dir}/train.csv')
    val_df = pd.read_csv(f'{args.fold_dir}/val.csv')
    df = pd.concat([train_df, val_df], ignore_index=True)

    validate_cache(args.cache, df, max_print=args.max_print, limit=args.limit, dedupe=not args.no_dedupe)
