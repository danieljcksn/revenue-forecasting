"""IO especifico do TCC: leitura/escrita de previsoes e artefatos.

Centraliza nomenclatura canonica de arquivos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.config import PipelineConfig

# ---------- Forecasts (saida do notebook 03) -----------------------------


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
    """Caminho de tabela LaTeX (.tex) gerada para o TCC."""
    if not name.endswith(".tex"):
        name = f"{name}.tex"
    return cfg.tables_dir_abs / name


def figure_path(cfg: PipelineConfig, name: str) -> Path:
    """Caminho de figura PDF gerada para o TCC."""
    if not name.endswith(".pdf"):
        name = f"{name}.pdf"
    return cfg.figures_dir_abs / name


# ---------- Dados crus do siconfi (saida do siconfi-collector) -----------

# O TCC chama de "ISSQN"; o RREO-Anexo 03 (e portanto o monthly_revenue.csv)
# usa a coluna "iss". Mapeia o nome do tributo no .tcc-pipeline.json para o
# nome da coluna no CSV.
TRIBUTO_COLUMN = {"IPTU": "iptu", "ISSQN": "iss", "ITBI": "itbi"}


def tributo_column(tributo: str) -> str:
    """Retorna o nome da coluna no monthly_revenue.csv para um tributo do TCC."""
    return TRIBUTO_COLUMN.get(tributo, tributo.lower())


def load_monthly_series(cfg: PipelineConfig) -> pd.DataFrame:
    """Carrega o CSV consolidado mensal (saida de `siconfi transform-monthly`).

    Colunas: cod_ibge, entity_name, uf, year, month, month_name, date,
    e uma coluna por categoria de receita (iptu, iss, itbi, fpm, icms, ...).
    Use `tributo_column("ISSQN")` -> "iss" para mapear nomes do TCC para
    colunas do CSV. O DataFrame inclui todos os municipios coletados; filtre
    para `cfg.municipalities` no notebook.
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
