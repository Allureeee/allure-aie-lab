from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    """Небольшой учебный DataFrame для базовых тестов."""
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic() -> None:
    """Проверяем, что summarize_dataset и flatten_summary_for_print
    возвращают корректные размеры и базовые колонки.
    """
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags() -> None:
    """Проверяем missing_table и базовый диапазон quality_score,
    а также наличие новых флагов в словаре.
    """
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)

    # quality_score должен быть в [0, 1]
    assert 0.0 <= flags["quality_score"] <= 1.0

    # словарь флагов содержит ключи для новых эвристик
    assert "has_constant_columns" in flags
    assert "has_suspicious_id_duplicates" in flags
    assert "has_many_zero_values" in flags


def test_quality_flags_for_constant_zero_and_id_duplicates() -> None:
    """Тест на три новые эвристики:
    - постоянную колонку;
    - дубликаты идентификатора;
    - много нулей в числовой колонке.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],  # дубликат ID
            "const_col": [5, 5, 5, 5],  # постоянная колонка
            "value": [0, 0, 10, 0],  # 3 из 4 значений — ноль (75 %)
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    # Константные колонки
    assert flags["has_constant_columns"] is True
    assert "const_col" in flags["constant_columns"]

    # Подозрительные дубликаты идентификаторов
    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags["id_columns_with_duplicates"]

    # Много нулей в числовых колонках
    assert flags["has_many_zero_values"] is True
    assert "value" in flags["many_zero_value_columns"]
    assert flags["zero_shares"]["value"] >= 0.5


def test_zero_share_below_threshold_has_no_many_zero_flag() -> None:
    """ДОПОЛНИТЕЛЬНЫЙ тест по 2.3.3:
    проверяем, что если доля нулей меньше порога, флаг has_many_zero_values=False.
    """
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "value": [0, 1, 2, 3],  # 1 ноль из 4 (25 %), ниже порога 50 %
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    # Доля нулей посчитана корректно
    assert pytest.approx(flags["zero_shares"]["value"], rel=1e-6) == 0.25

    # Порог по нулям не превышен — флаг должен быть False,
    # и колонка не должна быть в many_zero_value_columns.
    assert flags["has_many_zero_values"] is False
    assert "value" not in flags["many_zero_value_columns"]


def test_correlation_and_top_categories() -> None:
    """Проверяем, что считаются корреляции и top-категории."""
    df = _sample_df()
    corr = correlation_matrix(df)

    # либо матрица не пустая, либо в ней есть колонка age
    assert corr.empty is False or "age" in corr.columns

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2
