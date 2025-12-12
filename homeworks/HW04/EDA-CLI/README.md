# S04 – eda_cli: HTTP-сервис качества датасетов (FastAPI)

Расширенная версия проекта eda-cli из Семинара 03.

К существующему CLI-приложению для EDA добавлен HTTP-сервис на FastAPI с эндпоинтами /health, /quality и /quality-from-csv.
Используется в рамках Семинара 04 курса «Инженерия ИИ».

## Связь с S03

Проект в S04 основан на том же пакете eda_cli, что и в S03:

- сохраняется структура src/eda_cli/ и CLI-команда eda-cli;
- добавлен модуль api.py с FastAPI-приложением;
- в зависимости добавлены fastapi и uvicorn[standard].

Цель S04 – показать, как поверх уже написанного EDA-ядра поднять простой HTTP-сервис.

## Требования

- Python 3.11+
- uv установлен в систему
- Браузер (для Swagger UI /docs) или любой HTTP-клиент:
  - curl / HTTP-клиент в IDE / Postman / Hoppscotch и т.п.

## Инициализация проекта

В корне проекта (каталог S04/eda-cli):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml` (включая FastAPI и Uvicorn);
- установит сам проект `eda-cli` в окружение.

## Запуск CLI (как в S03)

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
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
  --out-dir reports \
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
- `bar_country.png` (или другой файл вида `bar_<колонка>.png`) — bar-chart по количеству объектов в выбранной категориальной колонке (дополнительное задание).

## Тесты

```bash
uv run pytest -q
```

## Запуск HTTP-сервиса

HTTP-сервис реализован в модуле `eda_cli.api` на FastAPI.

### Запуск Uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Пояснения:

- `eda_cli.api:app` - путь до объекта FastAPI `app` в модуле `eda_cli.api`;
- `--reload` - автоматический перезапуск сервера при изменении кода (удобно для разработки);
- `--port 8000` - порт сервиса (можно поменять при необходимости).

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

---

## Эндпоинты сервиса

### 1. `GET /health`

Простейший health-check.

**Запрос:**

```http
GET /health
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

Пример проверки через `curl`:

```bash
curl http://127.0.0.1:8000/health
```

---

### 2. Swagger UI: `GET /docs`

Интерфейс документации и тестирования API:

```text
http://127.0.0.1:8000/docs
```

Через `/docs` можно:

- вызывать `GET /health`;
- вызывать `POST /quality` (форма для JSON);
- вызывать `POST /quality-from-csv` (форма для загрузки файла).

---

### 3. `POST /quality` – заглушка по агрегированным признакам

Эндпоинт принимает **агрегированные признаки датасета** (размеры, доля пропусков и т.п.) и возвращает эвристическую оценку качества.

**Пример запроса:**

```http
POST /quality
Content-Type: application/json
```

Тело:

```json
{
  "n_rows": 10000,
  "n_cols": 12,
  "max_missing_share": 0.15,
  "numeric_cols": 8,
  "categorical_cols": 4
}
```

**Пример ответа `200 OK`:**

```json
{
  "ok_for_model": true,
  "quality_score": 0.8,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 3.2,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 10000,
    "n_cols": 12
  }
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "n_cols": 12, "max_missing_share": 0.15, "numeric_cols": 8, "categorical_cols": 4}'
```

---

### 4. `POST /quality-from-csv` – оценка качества по CSV-файлу

Эндпоинт принимает CSV-файл, внутри:

- читает его в `pandas.DataFrame`;
- вызывает функции из `eda_cli.core`:

  - `summarize_dataset`,
  - `missing_table`,
  - `compute_quality_flags`;
- возвращает оценку качества датасета в том же формате, что `/quality`.

**Запрос:**

```http
POST /quality-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

Через Swagger:

- в `/docs` открыть `POST /quality-from-csv`,
- нажать `Try it out`,
- выбрать файл (например, `data/example.csv`),
- нажать `Execute`.

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X POST "http://127.0.0.1:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

Ответ будет содержать:

- `ok_for_model` - результат по эвристикам;
- `quality_score` - интегральный скор качества;
- `flags` - булевы флаги из `compute_quality_flags`;
- `dataset_shape` - реальные размеры датасета (`n_rows`, `n_cols`);
- `latency_ms` - время обработки запроса.

## 5. `POST /quality-flags-from-csv (новый эндпоинт)`
Эндпоинт возвращает полный набор флагов качества, вычисленных функцией `compute_quality_flags`, включая новые эвристики из HW03:
- constant_columns
- has_constant_columns
- has many_zero_values
- has_suspicious_id_duplicates
- id_columns / id_columns_with_duplicates
- many_zero_value_columns
- max_missing_share
- quality_score
- too_few_rows
- too_many_columns
- too_many_missing
- zero_shares

**Запрос:**

```http
POST /quality-flags-from-csv
Content-Type: multipart/form-data
file: <CSV-file>
```
Через Swagger:

- в `/docs` открыть `POST /quality-flags-from-csv`,
- нажать `Try it out`,
- выбрать файл (например, `data/example.csv`),
- нажать `Execute`.

**Пример ответа:**

```http
{
  "flags": {
    "too_few_rows": False,
    "too_many_columns": False,
    "max_missing_share": 0.12,
    "too_many_missing": False,
    "constant_columns": [],
    "has_constant_columns": False,
    "id_columns": ["user_id"],
    "id_columns_with_duplicates": [],
    "has_suspicious_id_duplicates": False,
    "zero_shares": {"amount": 0.45},
    "many_zero_value_columns": [],
    "has_many_zero_values": False,
    "quality_score": 0.84,
  }
}
```
Этот эндпоинт используется, если требуется полный набор эвристик качества, а не только булевы признаки.

---
## `GET /metric (дополнительное задание)`
Возвращает статистику работы сервиса. Хранится в памяти и обновляется при вызове всех quality‑эндпоинтов.

Содержит:
- total_requests — общее число запросов ко всем quality-эндпоинтам;
- avg_latency_ms — средняя задержка обработки запроса;
- last_ok_for_model — последнее значение флага готовности датасета.

**Запрос:**

```http
GET /metrics
```

**Ожидаемый ответ `200 OK` (JSON):**

```http
{
"total_requests": 12,
"avg_latency_ms": 5.3,
"last_ok_for_model": true
}
```

Пример проверки через `curl`:

```bash
curl http://127.0.0.1:8000/metrics
```

Эндпоинт удобен для наблюдения за поведением сервиса и проверки нагрузки.

## Структура проекта (упрощённо)

```text
S04/
  eda-cli/
    pyproject.toml
    README.md                # этот файл
    src/
      eda_cli/
        __init__.py
        core.py              # EDA-логика, эвристики качества
        viz.py               # визуализации
        cli.py               # CLI (overview/report)
        api.py               # HTTP-сервис (FastAPI)
    tests/
      test_core.py           # тесты ядра
    data/
      example.csv            # учебный CSV для экспериментов
```

---

## Тесты

Запуск тестов (как и в S03):

```bash
uv run pytest -q
```

Рекомендуется перед любыми изменениями в логике качества данных и API:

1. Запустить тесты `pytest`;
2. Проверить работу CLI (`uv run eda-cli ...`);
3. Проверить работу HTTP-сервиса (`uv run uvicorn ...`, затем `/health` и `/quality`/`/quality-from-csv` через `/docs` или HTTP-клиент).