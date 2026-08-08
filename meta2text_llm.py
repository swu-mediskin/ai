import os
import json
import time
import base64
import mimetypes

import pandas as pd
from dotenv import load_dotenv

from meta2text import row2text, row2meta_dict, CSV_PATH

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_PATH = os.path.join(CACHE_DIR, 'new_llm_text_cache2.json')
VISION_CACHE_PATH = os.path.join(CACHE_DIR, 'new_llm_vision_text_cache.json')

LLM_MODEL = os.environ.get('LLM_META_MODEL', 'gpt-5.4-mini')
IMAGE_DETAIL = os.environ.get('LLM_IMAGE_DETAIL', 'low')

SYSTEM_PROMPT = """
<role>
당신은 PubMed의 케이스 리포트를 작성하는 피부과 전문의입니다.
</role>

<input>
아래 사용자 메시지로 환자/병변의 구조화된 raw 메타데이터가 JSON으로 주어집니다.
필드 의미: age(나이), gender(성별), region(병변 부위), fitzpatrick_skin_type(Fitzpatrick 피부타입, 1~6),
diameter_1_mm(병변의 수평 크기, mm), diameter_2_mm(병변의 수직 크기, mm), smoker(흡연 여부), drinks_alcohol(음주 여부),
pesticide_exposure(살충제를 일상에서 사용하는지 여부), skin_cancer_history(피부암 병력),
cancer_history(피부암 외 전신 암 병력),
itching/growing/hurting/changing/bleeding/elevating(각 병변 증상의 유무, true/false). 
</input>

<task>
주어진 메타데이터만 근거로, 실제 출판된 피부과 케이스 리포트와 같이 의학적 용어가 포함된 전문적인 문체로 서술을 작성하세요.
</task>

<constraints>
- 메타데이터에(JSON) 명시된 것만 서술하고, 없는 사실은 절대 추가하지 마세요.
- 진단명이나 악성/양성 여부를 언급하거나 암시하지 마세요. 
- 입력 메타데이터로 뒷받침되지 않는 사실은 절대 추가하지 마세요.
- [키가 아예 없는 경우] 입력 JSON에 아예 존재하지 않는 키(필드)에 대해서는 
  "없다/해당 없다/기록 없음"과 같은 부정문을 포함해 어떤 형태로든 절대 언급하지 마세요. 
  그 필드 자체가 존재하지 않는 것처럼 완전히 침묵하세요.
- [키는 있지만 값이 false인 경우] 입력 JSON에 키가 존재하고 그 값이 false인 경우에는, 
  이는 "해당 소견이 없음을 확인함"을 의미하므로 반드시 문장에 명시적으로 포함시켜야 합니다.
- 즉, "필드가 아예 없음"과 "필드는 있지만 값이 false임"은 반드시 구분해야 합니다.
</constraints>

<output_format>
- 반드시 영어로 작성하세요.
- 동일한 의학 용어를 반복하기보다, 문맥에 맞는 동의어나 대체 표현을 적절히 활용하세요 
- 순수 서술형 문단만 작성하세요. 
- 마크다운 헤더(#, ##, ###), 리스트, 굵은 글씨 등의 포맷팅 요소를 절대 포함하지 마세요.
</output_format>"""


VISION_SYSTEM_PROMPT = """<role>
당신은 PubMed 케이스 리포트의 'Case Presentation' 섹션을 작성하는 피부과 전문의입니다.
</role>

<input>
환자/병변의 구조화된 raw 메타데이터(JSON)와 병변 사진을 함께 받습니다.
필드 의미: age(나이), gender(성별), region(병변 부위), fitzpatrick_skin_type(Fitzpatrick 피부타입, 1~6),
diameter_1_mm(병변의 수평 크기, mm), diameter_2_mm(병변의 수직 크기, mm), smoker(흡연 여부), drinks_alcohol(음주 여부),
pesticide_exposure(살충제를 일상에서 사용하는지 여부), skin_cancer_history(피부암 병력),
cancer_history(피부암 외 전신 암 병력),
itching/growing/hurting/changing/bleeding/elevating(각 병변 증상의 유무, true/false). 
</input>

<task>
메타데이터와 사진을 종합해서, 실제 출판된 피부과 케이스 리포트와 같이 의학적 용어가 포함된 전문적인 문체로 서술을 작성하세요.
</task>

<visual_findings>
사진에서 실제로 관찰되는 객관적 시각 소견(병변의 형태, 색상, 경계의 규칙성, 표면 질감, 융기 정도 등)을
메타데이터의 병력·증상과 함께 반드시 포함하세요.
</visual_findings>

<constraints>
- 사진에서 객관적으로 보이는 것과 메타데이터에(JSON) 명시된 것만 서술하고, 없는 사실은 절대 추가하지 마세요.
- 진단명이나 악성/양성 여부를 언급하거나 암시하지 마세요. 
- 사진이나 입력 메타데이터로 뒷받침되지 않는 사실은 절대 추가하지 마세요.
- [키가 아예 없는 경우] 입력 JSON에 아예 존재하지 않는 키(필드)에 대해서는 
  "없다/해당 없다/기록 없음"과 같은 부정문을 포함해 어떤 형태로든 절대 언급하지 마세요. 
  그 필드 자체가 존재하지 않는 것처럼 완전히 침묵하세요.
- [키는 있지만 값이 false인 경우] 입력 JSON에 키가 존재하고 그 값이 false인 경우에는, 
  이는 "해당 소견이 없음을 확인함"을 의미하므로 반드시 문장에 명시적으로 포함시켜야 합니다.
- 즉, "필드가 아예 없음"과 "필드는 있지만 값이 false임"은 반드시 구분해야 합니다.
</constraints>

<output_format>
- 반드시 영어로 작성하세요.
- 동일한 의학 용어를 반복하기보다, 문맥에 맞는 동의어나 대체 표현을 적절히 활용하세요 
- 순수 서술형 문단만 작성하세요. 
- 마크다운 헤더(#, ##, ###), 리스트, 굵은 글씨 등의 포맷팅 요소를 절대 포함하지 마세요.
</output_format>"""

_caches = {}


def _load_cache(cache_path=CACHE_PATH):
    if cache_path not in _caches:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                _caches[cache_path] = json.load(f)
        else:
            _caches[cache_path] = {}
    return _caches[cache_path]


def _save_cache(cache, cache_path=CACHE_PATH):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _encode_image_data_uri(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or 'image/png'
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f"data:{mime_type};base64,{b64}"


def _get_client():
    from openai import OpenAI
    return OpenAI()


def _check_api_key():
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. "
            "GPT 기반 메타데이터 확장을 사용하려면 `export OPENAI_API_KEY=...`를 먼저 실행하세요."
        )


def _call_llm(base_text, image_path=None, model=LLM_MODEL, max_retries=3):
    image_data_uri = None
    if image_path is not None:
        try:
            image_data_uri = _encode_image_data_uri(image_path)
        except Exception as e:
            print(f"[meta2text_llm] image encoding failed for {image_path}: {e} -- falling back to text-only prompt")

    if image_data_uri is not None:
        system_prompt = VISION_SYSTEM_PROMPT
        user_content = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": image_data_uri, "detail": IMAGE_DETAIL}},
        ]
    else:
        system_prompt = SYSTEM_PROMPT
        user_content = base_text

    client = _get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
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


def row2text_llm(row, img_dir=None, use_cache=True):
    """row의 raw 메타데이터(JSON)를 GPT로 임상 서술형 문장으로 확장.

    img_dir이 주어지면 해당 폴더의 img_id 이미지를 함께 GPT에 전달해(vision) 사진 소견까지
    반영한 문장을 만든다. img_dir이 None이면 기존과 동일하게 메타데이터만 사용한다.
    두 모드는 서로 다른 캐시 파일을 사용한다. 실패 시 row2text() 템플릿 문장으로 폴백.
    """
    base_text = json.dumps(row2meta_dict(row), ensure_ascii=False)
    key = str(row.get('img_id', base_text))

    if img_dir is not None:
        image_path = os.path.join(img_dir, row['img_id'])
        cache_path = VISION_CACHE_PATH
    else:
        image_path = None
        cache_path = CACHE_PATH

    cache = _load_cache(cache_path) if use_cache else {}
    if use_cache and key in cache:
        return cache[key]

    try:
        expanded = _call_llm(base_text, image_path=image_path)
    except Exception as e:
        print(f"[meta2text_llm] LLM expansion failed for {key}: {e} -- falling back to template text")
        expanded = row2text(row)

    if use_cache:
        cache[key] = expanded
        _save_cache(cache, cache_path)
    return expanded


def build_llm_text_dict(df, img_dir=None, cache_path=None, save_every=20):
    """df(한 fold의 train/val csv)의 각 row에 대해 img_id -> GPT 확장 문장 dict를 만든다.

    img_dir이 주어지면 이미지를 함께 GPT에 전달하는 vision 모드로 동작하며, 기본 캐시 파일도
    VISION_CACHE_PATH로 바뀐다 (cache_path를 명시하면 그 값이 우선한다).
    이미 캐시(cache_path)에 있는 img_id는 API를 다시 호출하지 않는다.
    """
    _check_api_key()

    if cache_path is None:
        cache_path = VISION_CACHE_PATH if img_dir is not None else CACHE_PATH

    cache = _load_cache(cache_path)

    result = {}
    n_new = 0
    for _, row in df.iterrows():
        img_id = row['img_id']
        if img_id in cache:
            result[img_id] = cache[img_id]
            continue

        base_text = json.dumps(row2meta_dict(row), ensure_ascii=False)
        image_path = os.path.join(img_dir, img_id) if img_dir is not None else None
        try:
            expanded = _call_llm(base_text, image_path=image_path)
        except Exception as e:
            print(f"[meta2text_llm] LLM expansion failed for {img_id}: {e} -- falling back to template text")
            expanded = row2text(row)

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
    """metadata.csv 앞 5개 행에 대해 템플릿 문장, GPT 메타데이터 전용 문장,
    GPT 메타데이터+이미지 문장을 나란히 비교 출력."""
    _check_api_key()

    img_dir = os.path.join(CACHE_DIR, 'imgs')

    df = pd.read_csv(CSV_PATH).head(5)
    for _, row in df.iterrows():
        print(f"=== img_id: {row.get('img_id')} ===")
        print(f"[template     ] {row2text(row)}")
        print(f"[raw-meta     ] {json.dumps(row2meta_dict(row), ensure_ascii=False)}")
        print(f"[gpt-meta     ] {row2text_llm(row)}")
        print(f"[gpt-meta+img ] {row2text_llm(row, img_dir=img_dir)}")
        print()


if __name__ == "__main__":
    main()
