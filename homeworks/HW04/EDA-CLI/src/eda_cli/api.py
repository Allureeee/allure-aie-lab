from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Глобальные метрики ----------
metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "last_ok_for_model": None,
}


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    categorical_cols: int = Field(..., ge=0)


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    message: str
    latency_ms: float
    flags: dict[str, bool] | None = None
    dataset_shape: dict[str, int] | None = None


class QualityFlagsResponse(BaseModel):
    flags: dict = Field(
        ..., description="Полный набор флагов качества из compute_quality_flags",
        example={
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
        },
    )


class MetricsResponse(BaseModel):
    total_requests: int = Field(..., example=5)
    avg_latency_ms: float = Field(..., example=11.3)
    last_ok_for_model: bool | None = Field(None, example=True)


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dataset-quality", "version": "0.2.0"}


# ---------- /quality ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    start = perf_counter()

    score = 1.0
    score -= req.max_missing_share

    if req.n_rows < 1000:
        score -= 0.2

    if req.n_cols > 100:
        score -= 0.1

    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    message = (
        "Данных достаточно, модель можно обучать." if ok_for_model else
        "Качество данных недостаточно, требуется доработка."
    )

    latency_ms = (perf_counter() - start) * 1000

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Обновление метрик
    metrics["total_requests"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["last_ok_for_model"] = ok_for_model

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv ----------


@app.post("/quality-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл.")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл пуст.")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df=df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    message = (
        "CSV выглядит достаточно качественным." if ok_for_model else
        "CSV требует доработки."
    )

    latency_ms = (perf_counter() - start) * 1000

    flags_bool = {k: bool(v) for k, v in flags_all.items() if isinstance(v, bool)}

    n_rows = summary.n_rows
    n_cols = summary.n_cols

    # Метрики
    metrics["total_requests"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["last_ok_for_model"] = ok_for_model

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv ----------


@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse, tags=["quality"])
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    start = perf_counter()
    
    """
    Возвращает полный набор флагов качества из compute_quality_flags
    для загруженного CSV-файла.
    """
    # Проверка формата файла
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл.")

    # Чтение CSV
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл пуст.")

    # Вызов EDA-ядра
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df)

    latency_ms = (perf_counter() - start) * 1000

    # Метрики
    metrics["total_requests"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["last_ok_for_model"] = flags.get("quality_score", 0) >= 0.7

    # Возврат всех флагов
    return QualityFlagsResponse(flags=flags)


# ---------- /metrics ----------


@app.get("/metrics", response_model=MetricsResponse, tags=["system"])
def get_metrics() -> MetricsResponse:
    if metrics["total_requests"] > 0:
        avg_latency = metrics["total_latency_ms"] / metrics["total_requests"]
    else:
        avg_latency = 0.0

    return MetricsResponse(
        total_requests=metrics["total_requests"],
        avg_latency_ms=avg_latency,
        last_ok_for_model=metrics["last_ok_for_model"],
    )
