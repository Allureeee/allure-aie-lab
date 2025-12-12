from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Строит гистограммы для числовых колонок и сохраняет их как PNG.

    Возвращает список путей к сохранённым файлам.
    """
    out_dir = _ensure_dir(out_dir)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return []

    cols = list(numeric.columns[:max_columns])
    paths: List[Path] = []

    for col in cols:
        series = numeric[col].dropna()
        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=bins)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        fig.tight_layout()
        out_path = out_dir / f"hist_{col}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Визуализация пропусков: матрица (строки — объекты, столбцы — признаки).
    """
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    if df.empty:
        # создадим пустой рисунок, чтобы файл всё равно был
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title("No data")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    mask = df.isna().values
    fig, ax = plt.subplots(figsize=(max(4, df.shape[1] * 0.4), 4))
    ax.imshow(mask, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_title("Missing values matrix")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляций для числовых признаков.
    """
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        # Слишком мало колонок – сохраняем «заглушку»
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title("Not enough numeric columns for correlation")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(4, corr.shape[1] * 0.6), 4))
    im = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_category_counts_bar(
    df: pd.DataFrame,
    column: str,
    out_path: PathLike,
    top_k: int | None = None,
) -> Path:
    """
    Bar-chart по количеству объектов в каждой категории указанной колонки.

    - column: имя категориальной колонки;
    - top_k: если задано, берём только top_k наиболее частых значений.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    counts = df[column].value_counts(dropna=False)
    if top_k is not None:
        counts = counts.head(top_k)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Counts by '{column}'")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)
    return paths
