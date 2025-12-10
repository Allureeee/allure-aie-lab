from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
    plot_category_counts_bar,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6,
        help="Максимум числовых колонок, для которых строятся гистограммы.",
    ),
    top_k_categories: int = typer.Option(
        5,
        help="Сколько top-значений сохранять для каждой категориальной колонки.",
    ),
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков; колонки с большей долей считаются проблемными.",
    ),
    title: Optional[str] = typer.Option(
        None,
        help="Заголовок отчёта (по умолчанию 'EDA-отчёт').",
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор и табличные данные
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество данных
    quality_flags = compute_quality_flags(summary, missing_df, df=df)

    # Колонки с долей пропусков выше порога
    problematic_missing_columns: list[str] = []
    if not missing_df.empty:
        problematic = missing_df[missing_df["missing_share"] >= min_missing_share]
        problematic_missing_columns = problematic.index.tolist()

    # 3. Bar-chart по категориальной колонке
    bar_chart_path: Optional[Path] = None
    bar_chart_column: Optional[str] = None

    # предпочтительно использовать колонку country, если она есть;
    # иначе берём первую строковую/категориальную колонку.
    if "country" in df.columns:
        bar_chart_column = "country"
    else:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            bar_chart_column = str(cat_cols[0])

    if bar_chart_column is not None:
        bar_chart_path = plot_category_counts_bar(
            df,
            column=bar_chart_column,
            out_path=out_root / f"bar_{bar_chart_column}.png",
            top_k=top_k_categories,
        )

    # 4. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    report_title = title or "EDA-отчёт"

    constant_cols_display = ", ".join(quality_flags["constant_columns"]) or "нет"
    id_dups_display = ", ".join(quality_flags["id_columns_with_duplicates"]) or "нет"
    many_zero_display = ", ".join(quality_flags["many_zero_value_columns"]) or "нет"

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {report_title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Постоянные колонки: {constant_cols_display}\n")
        f.write(f"- ID с дубликатами: {id_dups_display}\n")
        f.write(f"- Колонки с долей нулей ≥ 50%: {many_zero_display}\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        f.write(
            f"Порог для проблемных колонок по пропускам: **{min_missing_share:.0%}**.\n\n"
        )
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")
            if problematic_missing_columns:
                f.write("Колонки с долей пропусков выше порога:\n")
                for col in problematic_missing_columns:
                    share = float(missing_df.loc[col, "missing_share"])
                    f.write(f"- {col}: {share:.2%}\n")
                f.write("\n")
            else:
                f.write("Нет колонок с долей пропусков выше заданного порога.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(
                f"Для каждой категориальной колонки сохранены top-{top_k_categories} "
                "значений в каталоге `top_categories/`.\n\n"
            )

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(
            f"Построены гистограммы максимум для {max_hist_columns} числовых колонок.\n"
        )
        f.write("См. файлы `hist_*.png`.\n\n")

        f.write("## Распределение по категориям\n\n")
        if bar_chart_path is not None and bar_chart_column is not None:
            f.write(
                "Построен bar-chart по количеству объектов в категориальной колонке "
                f"`{bar_chart_column}`. См. файл `{bar_chart_path.name}`.\n\n"
            )
        else:
            f.write(
                "Категориальная колонка для построения bar-chart не найдена "
                "(нет колонок типа object/category).\n\n"
            )

    # 6. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    if bar_chart_path is not None:
        typer.echo(f"- Bar-chart категорий: {bar_chart_path}")


if __name__ == "__main__":
    app()
