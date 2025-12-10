# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports_example
```
Дополнительные параметры команды report:

- max-hist-columns – сколько числовых колонок включать в набор гистограмм (по умолчанию 6);

- top-k-categories – сколько top-значений сохранять для каждой категориальной колонки (по умолчанию 5);
- min-missing-share – порог доли пропусков, выше которого колонка считается проблемной и отдельно перечисляется в отчёте (по умолчанию 0.3 = 30%);
- title – заголовок отчёта (по умолчанию EDA-отчёт);
- sep, --encoding – как и в команде overview.

Пример вызова с опциями:
```bash
uv run eda-cli report data/example.csv \
  --out-dir reports_example \
  --max-hist-columns 8 \
  --top-k-categories 5 \
  --min-missing-share 0.4 \
  --title "EDA для example.csv"
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
