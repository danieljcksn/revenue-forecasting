# -*- coding: utf-8 -*-
"""Benchmark anual em todos os municipios baianos coletados via SICONFI.

Analise independente do TCC (nao altera nenhum artefato do texto): aplica o
MESMO metodo do trabalho -- deflacao IPCA, controle de qualidade de
``forecasting.generalization.prepare_extended_series`` e o portfolio canonico
(Naive sazonal, AutoETS, AutoARIMA sobre log, AutoTheta, Prophet, Ensemble) --
restrito as ORIGENS DE DEZEMBRO, que alimentam o unico confronto de interesse
aqui: a previsao anual do modelo contra a "Previsao Atualizada" registrada pela
propria prefeitura no RREO-Anexo 03.

Em cada origem (dezembro do ano Y), cada modelo e reestimado do zero sobre o
treino ate a origem e emite doze previsoes mensais; a soma e a previsao do
exercicio Y+1. O protocolo espelha a Secao 5.7 do TCC (sem correcao de Jensen,
como la: ponto mensal do SARIMA = exponencial da previsao em log).

Uso (no venv com statsforecast + prophet):
    python benchmark.py [--workers N] [--limit N]

Le    : siconfi-collector/data/transformed/monthly_revenue.csv (via pacote)
Escreve: out/cv_bahia.csv  (cod_ibge, municipio, tributo, modelo, target_year,
                            pred_annual, real_annual)
         out/qc_log.txt    (aprovadas, imputadas, interpoladas, excluidas)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../analysis/bahia
ANALYSIS = HERE.parent                          # .../analysis
sys.path.insert(0, str(ANALYSIS / "src"))
OUT = HERE / "out"

CFG_PATH = None  # usa o .tcc-pipeline.json da raiz do repo
INITIAL_WINDOW = 72     # seis ciclos sazonais, identico ao TCC
SEASON = 12
HORIZON = 12
FORMAL = ["ETS", "SARIMA", "Theta", "Prophet"]

warnings.filterwarnings("ignore")


# ---------- preparo (processo principal) -----------------------------------


def prepared_series():
    """Series deflacionadas e aprovadas no QC, para TODOS os municipios
    presentes na coleta (nao apenas os populosos)."""
    from forecasting.config import load_config
    from forecasting.generalization import prepare_extended_series
    from forecasting.io import load_monthly_series

    cfg = load_config(CFG_PATH)
    raw = load_monthly_series(cfg)
    munis = (raw[["cod_ibge", "entity_name"]].drop_duplicates()
             .assign(nome=lambda d: d["entity_name"].astype(str)
                     .str.replace("Prefeitura Municipal de ", "", regex=False)
                     .str.replace(" - BA", "", regex=False).str.strip()))
    lista = [(int(r.cod_ibge), r.nome) for r in munis.itertuples()]
    series, log = prepare_extended_series(cfg, municipios=lista)
    return series, log, len(lista)


def december_origins(s: pd.Series) -> list[int]:
    """Fins de treino em dezembro com doze meses observados a frente."""
    return [i for i in range(INITIAL_WINDOW, len(s))
            if s.index[i - 1].month == 12 and i + HORIZON <= len(s)]


# ---------- ajuste (workers) ------------------------------------------------


def _forecast_paths(train: np.ndarray, train_index: pd.DatetimeIndex) -> dict:
    """Doze passos de cada previsor do portfolio, treinados em ``train``.

    Naive: repete o ultimo ciclo sazonal. SARIMA: ajustado sobre log, volta
    pela exponencial (mediana). Ensemble: media simples dos quatro formais.
    """
    from prophet import Prophet
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    paths: dict[str, np.ndarray] = {}
    paths["Naive"] = train[-SEASON:].astype(float).copy()

    ets = AutoETS(season_length=SEASON, model="ZZZ").fit(train)
    paths["ETS"] = np.asarray(ets.predict(h=HORIZON)["mean"], dtype=float)

    sar = AutoARIMA(season_length=SEASON).fit(np.log(train))
    paths["SARIMA"] = np.exp(np.asarray(sar.predict(h=HORIZON)["mean"],
                                        dtype=float))

    theta = AutoTheta(season_length=SEASON).fit(train)
    paths["Theta"] = np.asarray(theta.predict(h=HORIZON)["mean"], dtype=float)

    m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                yearly_seasonality=6, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05, changepoint_range=0.8)
    m.fit(pd.DataFrame({"ds": train_index, "y": train}))
    future = pd.DataFrame({"ds": pd.date_range(
        train_index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")})
    paths["Prophet"] = m.predict(future)["yhat"].to_numpy(dtype=float)

    paths["Ensemble"] = np.mean([paths[k] for k in FORMAL], axis=0)
    return paths


def run_series(payload) -> tuple[tuple, list[dict], str | None]:
    """Executa uma serie inteira (todas as origens de dezembro) num worker."""
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)

    (cod, nome, tributo), values, index = payload
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(index))
    rows: list[dict] = []
    try:
        for oi in december_origins(s):
            train = s.iloc[:oi]
            target_year = int(s.index[oi].year)
            real_annual = float(s.iloc[oi:oi + HORIZON].sum())
            paths = _forecast_paths(train.to_numpy(dtype=float), train.index)
            for modelo, path in paths.items():
                if not np.isfinite(path).all():
                    raise ValueError(f"previsao nao-finita ({modelo})")
                rows.append(dict(cod_ibge=cod, municipio=nome, tributo=tributo,
                                 modelo=modelo, target_year=target_year,
                                 pred_annual=float(path.sum()),
                                 real_annual=real_annual))
        return (cod, nome, tributo), rows, None
    except Exception as exc:                       # noqa: BLE001
        return (cod, nome, tributo), [], f"{type(exc).__name__}: {exc}"


# ---------- orquestracao ----------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="processa apenas N series (validacao rapida)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    series, log, n_munis = prepared_series()
    print(f"municipios na coleta: {n_munis} | series aprovadas no QC: "
          f"{len(series)}", flush=True)

    keys = sorted(series.keys(), key=lambda k: (k[1], k[2]))
    if args.limit:
        keys = keys[:args.limit]
    payloads = [((cod, nome, trib),
                 series[(cod, nome, trib)].to_numpy(dtype=float),
                 series[(cod, nome, trib)].index)
                for (cod, nome, trib) in keys]

    t0 = time.time()
    frames: list[pd.DataFrame] = []
    fails: list[tuple] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_series, p): p[0] for p in payloads}
        for fut in as_completed(futures):
            key, rows, err = fut.result()
            done += 1
            if err:
                fails.append((key[1], key[2], err))
                print(f"[{done}/{len(payloads)}] SKIP {key[1]} {key[2]}: {err}",
                      flush=True)
            else:
                frames.append(pd.DataFrame(rows))
                if done % 25 == 0 or done == len(payloads):
                    rate = done / (time.time() - t0)
                    eta = (len(payloads) - done) / rate / 60
                    print(f"[{done}/{len(payloads)}] ok "
                          f"({rate:.1f} series/s, ETA {eta:.0f} min)", flush=True)

    cv = pd.concat(frames, ignore_index=True)
    cv.to_csv(OUT / "cv_bahia.csv", index=False, encoding="utf-8")
    (OUT / "qc_log.txt").write_text(
        f"MUNICIPIOS NA COLETA: {n_munis}\n"
        f"SERIES APROVADAS NO QC: {len(series)}\n"
        f"SERIES MODELADAS OK: {len(frames)}\n"
        f"imputed: {log['imputed']}\n"
        f"interpolated: {log['interpolated']}\n"
        f"excluded: {log['excluded']}\n"
        f"absent: {log['absent']}\n"
        f"fit_failed: {fails}\n", encoding="utf-8")
    print(f"\nDONE: {len(frames)} series, {len(cv)} linhas em "
          f"{(time.time()-t0)/60:.1f} min -> {OUT/'cv_bahia.csv'}", flush=True)


if __name__ == "__main__":
    main()
