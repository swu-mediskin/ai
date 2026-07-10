import os
import json
import time

import pandas as pd
from dotenv import load_dotenv

from meta2text import row2text, CSV_PATH

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_PATH = os.path.join(CACHE_DIR, 'llm_text_cache.json')

LLM_MODEL = os.environ.get('LLM_META_MODEL', 'gpt-4o-mini')

SYSTEM_PROMPT = (
    "You are a dermatologist writing the 'Case Presentation' section of a PubMed case report. "
    "Rewrite the structured patient/lesion summary below as 2-4 sentences (under 100 words) "
    "in the dense, formal register of a published dermatology case report -- not a plain "
    "paraphrase of the input sentences. "
    "Use precise clinical/medical terminology instead of lay wording, for example: "
    "'pruritus' instead of itching, 'cutaneous lesion' instead of lesion, 'hemorrhage' or "
    "'bleeding on contact' instead of bleeding, 'progressive enlargement' instead of growing, "
    "'raised/elevated morphology' instead of elevation, 'significant smoking history' instead of "
    "is a smoker, 'notable for a prior history of skin cancer' instead of has a history of skin cancer. "
    "Frame it as a consultation note, e.g. starting with 'Clinical case report of a...', "
    "'Dermatological consultation reveals...', or 'Physical examination reveals...'. "
    "Do not state or imply a diagnosis, malignancy, or benign/malignant likelihood of any kind -- "
    "describe only the objective history and symptoms given. "
    "Do not add any fact that is not present in the input."
)

_cache = None


def _load_cache(cache_path=CACHE_PATH):
    global _cache
    if _cache is None:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save_cache(cache, cache_path=CACHE_PATH):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _get_client():
    from openai import OpenAI
    return OpenAI()


def _check_api_key():
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. "
            "GPT 기반 메타데이터 확장을 사용하려면 `export OPENAI_API_KEY=...`를 먼저 실행하세요."
        )


def _call_llm(base_text, model=LLM_MODEL, max_retries=3):
    client = _get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_text},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_err


def row2text_llm(row, use_cache=True):
    """row2text()의 결과를 GPT로 임상 서술형 문장으로 확장. 실패 시 row2text() 결과로 폴백."""
    base_text = row2text(row)
    key = str(row.get('img_id', base_text))

    cache = _load_cache() if use_cache else {}
    if use_cache and key in cache:
        return cache[key]

    try:
        expanded = _call_llm(base_text)
    except Exception as e:
        print(f"[meta2text_llm] LLM expansion failed for {key}: {e} -- falling back to template text")
        expanded = base_text

    if use_cache:
        cache[key] = expanded
        _save_cache(cache)
    return expanded


def build_llm_text_dict(df, cache_path=CACHE_PATH, save_every=20):
    """df(한 fold의 train/val csv)의 각 row에 대해 img_id -> GPT 확장 문장 dict를 만든다.

    이미 캐시(cache_path)에 있는 img_id는 API를 다시 호출하지 않는다.
    """
    _check_api_key()

    cache = _load_cache(cache_path)

    result = {}
    n_new = 0
    for _, row in df.iterrows():
        img_id = row['img_id']
        if img_id in cache:
            result[img_id] = cache[img_id]
            continue

        base_text = row2text(row)
        try:
            expanded = _call_llm(base_text)
        except Exception as e:
            print(f"[meta2text_llm] LLM expansion failed for {img_id}: {e} -- falling back to template text")
            expanded = base_text

        cache[img_id] = expanded
        result[img_id] = expanded
        n_new += 1
        if n_new % save_every == 0:
            _save_cache(cache, cache_path)
            print(f"[meta2text_llm] {n_new}건 확장 완료 (캐시 저장됨)")

    if n_new % save_every != 0:
        _save_cache(cache, cache_path)
    print(f"[meta2text_llm] 총 {len(result)}건 중 신규 {n_new}건 GPT 확장, {len(result) - n_new}건 캐시 재사용")

    return result


def main():
    """metadata.csv 앞 5개 행에 대해 템플릿 문장과 GPT 확장 문장을 비교 출력."""
    _check_api_key()

    df = pd.read_csv(CSV_PATH).head(5)
    for _, row in df.iterrows():
        print(f"=== img_id: {row.get('img_id')} ===")
        print(f"[template] {row2text(row)}")
        print(f"[gpt     ] {row2text_llm(row)}")
        print()


if __name__ == "__main__":
    main()
