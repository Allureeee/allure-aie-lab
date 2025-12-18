# EDA-отчёт

Исходный файл: `example.csv`

Строк: **36**, столбцов: **14**

## Качество данных (эвристики)

- quality_score: **0.59**
- max_missing_share: **5.56%**
- too_few_rows: **True**
- too_many_columns: **False**
- too_many_missing: **False**
- has_constant_columns: **False**
- constant_columns: нет
- has_suspicious_id_duplicates: **True**
- id_columns_with_duplicates: user_id
- has_many_zero_values: **True**
- many_zero_value_columns: churned

Доля нулей по колонкам (только для отмеченных):
- churned: 66.67%

## Колонки

См. файл `summary.csv`.

## Пропуски

Порог для проблемных колонок по пропускам: **30%**.

См. файлы `missing.csv` и `missing_matrix.png`.

Нет колонок с долей пропусков выше заданного порога.

## Корреляция числовых признаков

См. `correlation.csv` и `correlation_heatmap.png`.

## Категориальные признаки

Для каждой категориальной колонки сохранены top-5 значений в каталоге `top_categories/`.

## Гистограммы числовых колонок

Построены гистограммы максимум для 6 числовых колонок.
См. файлы `hist_*.png`.

## Распределение по категориям

Построен bar-chart по количеству объектов в категориальной колонке `country`. См. файл `bar_country.png`.

