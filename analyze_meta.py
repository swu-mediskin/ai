"""
data/metadata.csv의 각 컬럼과 diagnostic(NEV/BCC/SCC/MEL 4개 클래스) 간의 연관도를 분석한다.
- 수치형 컬럼(age, fitspatrick, diameter_1, diameter_2): one-way ANOVA + eta-squared
- 범주형 컬럼(gender, region, smoke 등 yes/no, 출신 지역 등): chi-square + Cramer's V
연관도 점수(eta-squared / Cramer's V) 기준으로 정렬해 터미널에 출력한다.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'metadata.csv')

TARGET_COL = 'diagnostic'
CLASSES = ['NEV', 'BCC', 'SCC', 'MEL']
EXCLUDE_COLS = ['patient_id', 'lesion_id', 'img_id', TARGET_COL]


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df[df[TARGET_COL].isin(CLASSES)].reset_index(drop=True)


def split_columns(df):
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def anova_eta_squared(df, col):
    groups = [series.dropna().values for _, series in df.groupby(TARGET_COL)[col]]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return None

    f_stat, p_value = stats.f_oneway(*groups)

    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = np.sum((all_values - grand_mean) ** 2)
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        'variable': col,
        'f_stat': f_stat,
        'p_value': p_value,
        'score': eta_sq,
        'n': len(all_values),
    }


def cramers_v(df, col):
    contingency = pd.crosstab(df[col], df[TARGET_COL])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    r, k = contingency.shape
    v = np.sqrt((chi2 / n) / min(r - 1, k - 1))

    return {
        'variable': col,
        'chi2': chi2,
        'p_value': p_value,
        'score': v,
        'dof': dof,
        'n': n,
    }


def print_numeric_table(results):
    print(f"{'변수':<22}{'F-statistic':>14}{'p-value':>14}{'eta^2':>10}{'표본수':>10}")
    print('-' * 70)
    for r in results:
        print(f"{r['variable']:<22}{r['f_stat']:>14.4f}{r['p_value']:>14.4g}"
              f"{r['score']:>10.4f}{r['n']:>10d}")


def print_categorical_table(results):
    print(f"{'변수':<22}{'chi-square':>14}{'p-value':>14}{'Cramer V':>11}{'dof':>6}{'표본수':>10}")
    print('-' * 70)
    for r in results:
        print(f"{r['variable']:<22}{r['chi2']:>14.4f}{r['p_value']:>14.4g}"
              f"{r['score']:>11.4f}{r['dof']:>6d}{r['n']:>10d}")


def main():
    df = load_data()

    print(f"분석 대상: {len(df)}개 샘플 (클래스: {', '.join(CLASSES)})")
    print(df[TARGET_COL].value_counts().reindex(CLASSES))
    print()

    numeric_cols, categorical_cols = split_columns(df)

    numeric_results = [r for r in (anova_eta_squared(df, c) for c in numeric_cols) if r]
    numeric_results.sort(key=lambda r: r['score'], reverse=True)

    categorical_results = [r for r in (cramers_v(df, c) for c in categorical_cols) if r]
    categorical_results.sort(key=lambda r: r['score'], reverse=True)

    print('=' * 70)
    print('[수치형] One-way ANOVA + eta-squared (연관도 점수, 클래스별 평균 차이의 효과크기)')
    print('=' * 70)
    print_numeric_table(numeric_results)

    print()
    print('=' * 70)
    print("[범주형] Chi-square + Cramer's V (연관도 점수, 0~1)")
    print('=' * 70)
    print_categorical_table(categorical_results)

    print()
    print('=' * 70)
    print('전체 변수 연관도 순위 (eta^2 / Cramer V 점수 기준 통합 정렬)')
    print('=' * 70)
    combined = (
        [('수치형', r['variable'], r['score']) for r in numeric_results]
        + [('범주형', r['variable'], r['score']) for r in categorical_results]
    )
    combined.sort(key=lambda x: x[2], reverse=True)
    for rank, (kind, name, score) in enumerate(combined, start=1):
        print(f"{rank:>2}. [{kind}] {name:<22} score={score:.4f}")


if __name__ == '__main__':
    main()
