"""Read and write forecast caches and generated artifacts.

Centraliza nomenclatura canonica de arquivos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.config import PipelineConfig

# ---------- Forecast files ------------------------------------------------


def forecast_path(
    cfg: PipelineConfig,
    municipio_key: str,
    tributo: str,
    modelo: str,
    horizon: int,
) -> Path:
    """Caminho canonico para previsoes de uma serie/modelo/horizonte."""
    return (
        cfg.forecasts_dir
        / f"{municipio_key}_{tributo.lower()}_{modelo.lower()}_h{horizon}.csv"
    )


def save_forecast(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    municipio_key: str,
    tributo: str,
    modelo: str,
    horizon: int,
) -> Path:
    """Grava DataFrame de previsoes em path canonico (CSV UTF-8 com BOM)."""
    path = forecast_path(cfg, municipio_key, tributo, modelo, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------- Convencoes para tabelas e figuras ----------------------------


def table_path(cfg: PipelineConfig, name: str) -> Path:
    """Path for a generated LaTeX table."""
    if not name.endswith(".tex"):
        name = f"{name}.tex"
    return cfg.tables_dir_abs / name


def figure_path(cfg: PipelineConfig, name: str) -> Path:
    """Path for a generated PDF figure."""
    if not name.endswith(".pdf"):
        name = f"{name}.pdf"
    return cfg.figures_dir_abs / name


# ---------- Collector outputs --------------------------------------------

# The collector stores ISSQN as the CSV column "iss"; analysis code uses the
# public tax label "ISSQN".
TRIBUTO_COLUMN = {"IPTU": "iptu", "ISSQN": "iss", "ITBI": "itbi"}


def tributo_column(tributo: str) -> str:
    """Return the monthly_revenue.csv column name for a tax label."""
    return TRIBUTO_COLUMN.get(tributo, tributo.lower())


def load_monthly_series(cfg: PipelineConfig) -> pd.DataFrame:
    """Carrega o CSV consolidado mensal (saida de `siconfi transform-monthly`).

    Colunas: cod_ibge, entity_name, uf, year, month, month_name, date,
    e uma coluna por categoria de receita (iptu, iss, itbi, fpm, icms, ...).
    Use `tributo_column("ISSQN")` -> "iss" to map public labels to CSV
    columns. The DataFrame includes all collected municipalities.
    """
    path = cfg.siconfi_data_dir / "transformed" / "monthly_revenue.csv"
    return pd.read_csv(path, parse_dates=["date"])


def load_prefeitura_forecast(cfg: PipelineConfig) -> pd.DataFrame:
    """Carrega tabela de previsoes da prefeitura (saida de
    `siconfi transform-prefeitura-forecast`).

    Colunas: cod_ibge, entity_name, year, tributo, previsao_prefeitura,
    periodo_fonte (1 = mais proximo da Previsao Inicial da LOA),
    realizado_anual, erro_pct_prefeitura, vies_prefeitura. O `tributo` ja
    vem normalizado como "IPTU"/"ISSQN"/"ITBI" (o ISS do Anexo 03 e
    renomeado para ISSQN na origem).
    """
    path = cfg.siconfi_data_dir / "transformed" / "prefeitura_forecast.csv"
    return pd.read_csv(path)
