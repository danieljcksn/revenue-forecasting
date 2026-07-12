# -*- coding: utf-8 -*-
"""Regenera o portfolio canonico de SEIS modelos (cv_all.csv + sarima_var.csv).

Consolidacao do antigo _precisao_run/ como pipeline oficial (goal C3).
O trabalho e dividido em estagios porque dois ambientes sao necessarios
(statsforecast/numba nao roda no Python do venv principal):

  --stage sf       AutoETS 'ZZZ' + AutoTheta        [venv precisao: requirements-sf-lock.txt]
  --stage classic  Naive + SARIMA D=1 (com mu/sigma p/ Jensen) + Prophet v2
                                                     [venv principal: requirements-lock.txt]
  --stage merge    monta cv_all_rebuild.csv (schema canonico) + sarima_var_rebuild.csv
  --stage verify   compara o rebuild com o cache canonico versionado e reporta
                   o desvio maximo; falha se exceder a tolerancia
  --promote        (apos verify ok) substitui cv_all.csv e sarima_var.csv pelo rebuild

Saidas intermediarias em data/forecasts/_rebuild/ (fora do git).
Sequencia completa:
  (venv sf)        python scripts/run_pipeline_full.py --stage sf
  (venv principal) python scripts/run_pipeline_full.py --stage classic
  (qualquer)       python scripts/run_pipeline_full.py --stage merge --stage verify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import load_config
from forecasting.eda import prepare_series
from forecasting.models import (
    INITIAL_WINDOW,
    MAX_HORIZON,
    ROLLING_STEP,
    fit_naive_seasonal,
    rolling_origin_cv,
)

MUN_NOME = {"salvador": "Salvador", "camacari": "Camaçari", "ilheus": "Ilhéus"}
CANON = ["municipio_nome", "municipio", "tributo", "modelo", "origin",
         "train_end", "step", "target_date", "y_true", "y_pred",
         "insample_scale", "regime"]
# Tolerancia relativa da equivalencia rebuild vs cache: acima disso o verify
# FALHA. Medido na consolidacao (10/07/2026): y_true e insample_scale exatos
# (0,0); y_pred com desvio maximo de 2,1e-5, concentrado no SARIMA (ruido de
# otimizador MLE do statsmodels entre execucoes). A invariancia dos numeros
# reportados foi provada a parte: scripts/check_winrates.py sobre o rebuild
# reproduz 16/16 numeros canonicos. 1e-4 cobre o ruido de otimizador com
# folga e ainda pega qualquer mudanca real de modelo/dado.
RTOL = 1e-4


def _rebuild_dir(cfg) -> Path:
    d = cfg.forecasts_dir / "_rebuild"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_sf(cfg) -> None:
    from forecasting.precisao import make_sf_specs, sf_rolling_cv
    out = _rebuild_dir(cfg)
    series = prepare_series(cfg, impute=True)
    for name, spec in make_sf_specs().items():
        frames = []
        for (mun, trib), s in series.items():
            t0 = time.perf_counter()
            d = sf_rolling_cv(s, spec)
            d["municipio"] = mun
            d["tributo"] = trib
            frames.append(d)
            print(f"  [{name}] {mun}-{trib:5s} rows={len(d):4d} "
                  f"{time.perf_counter() - t0:5.1f}s", flush=True)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / f"cv_sf_{name}.csv", index=False)
        print(f"=> {out / f'cv_sf_{name}.csv'} ({len(df)} linhas)", flush=True)


def stage_classic(cfg) -> None:
    from forecasting.precisao import (
        make_prophet_fitter,
        rolling_cv_any,
        sarima_D1_rolling_with_var,
    )
    out = _rebuild_dir(cfg)
    series = prepare_series(cfg, impute=True)

    frames = []
    for (mun, trib), s in series.items():
        d = rolling_origin_cv(s, fit_naive_seasonal, INITIAL_WINDOW,
                              MAX_HORIZON, ROLLING_STEP)
        d["municipio"] = mun
        d["tributo"] = trib
        frames.append(d)
    pd.concat(frames, ignore_index=True).to_csv(out / "cv_naive.csv", index=False)
    print("=> cv_naive.csv", flush=True)

    frames = []
    for (mun, trib), s in series.items():
        t0 = time.perf_counter()
        d = sarima_D1_rolling_with_var(s)
        order, sorder = d.attrs["selected_orders"]
        d["municipio"] = mun
        d["tributo"] = trib
        frames.append(d)
        print(f"  [SARIMA-D1] {mun}-{trib:5s} order={order} seas={sorder} "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "cv_sarima_var.csv",
                                                index=False)
    print("=> cv_sarima_var.csv", flush=True)

    prophet_fit = make_prophet_fitter(use_holidays=False, yearly_fourier=6,
                                      cps=0.05, cr=0.8)
    frames = []
    for (mun, trib), s in series.items():
        t0 = time.perf_counter()
        d = rolling_cv_any(s, prophet_fit)
        d["municipio"] = mun
        d["tributo"] = trib
        frames.append(d)
        print(f"  [Prophet] {mun}-{trib:5s} rows={len(d):4d} "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "cv_prophet.csv",
                                                index=False)
    print("=> cv_prophet.csv", flush=True)


def stage_merge(cfg) -> None:
    out = _rebuild_dir(cfg)

    def load(name, modelo):
        d = pd.read_csv(out / name,
                        parse_dates=["origin", "train_end", "target_date"])
        d["modelo"] = modelo
        return d

    naive = load("cv_naive.csv", "Naive")
    ets = load("cv_sf_ETS.csv", "ETS")
    theta = load("cv_sf_Theta.csv", "Theta")
    prophet = load("cv_prophet.csv", "Prophet")
    svar = pd.read_csv(out / "cv_sarima_var.csv",
                       parse_dates=["origin", "train_end", "target_date"])
    sarima = svar.rename(columns={"y_pred_median": "y_pred"})
    sarima["modelo"] = "SARIMA"

    cols = ["municipio", "tributo", "modelo", "origin", "train_end", "step",
            "target_date", "y_true", "y_pred", "insample_scale"]
    allm = pd.concat([d[cols] for d in (naive, ets, sarima, theta, prophet)],
                     ignore_index=True)

    formal = allm[allm["modelo"].isin(["ETS", "SARIMA", "Theta", "Prophet"])]
    ens = (formal.groupby(["municipio", "tributo", "origin", "train_end",
                           "step", "target_date"])
           .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                insample_scale=("insample_scale", "first")).reset_index())
    ens["modelo"] = "Ensemble"
    allm = pd.concat([allm, ens[cols]], ignore_index=True)

    allm["municipio_nome"] = allm["municipio"].map(MUN_NOME)

    def regime(d):
        d = pd.Timestamp(d)
        if d < pd.Timestamp("2020-03-01"):
            return "pre"
        if d <= pd.Timestamp("2021-12-31"):
            return "covid"
        return "pos"

    allm["regime"] = allm["target_date"].apply(regime)
    allm[CANON].to_csv(out / "cv_all_rebuild.csv", index=False)
    print(f"=> cv_all_rebuild.csv ({len(allm)} linhas)", flush=True)

    svar["serie"] = svar["municipio"] + "-" + svar["tributo"]
    var_cols = ["serie", "municipio", "tributo", "origin", "train_end", "step",
                "target_date", "y_true", "mu_log", "sigma_log",
                "y_pred_median", "y_pred_mean", "insample_scale"]
    svar[var_cols].to_csv(out / "sarima_var_rebuild.csv", index=False)
    print(f"=> sarima_var_rebuild.csv ({len(svar)} linhas)", flush=True)


def stage_verify(cfg) -> int:
    out = _rebuild_dir(cfg)
    keys = ["municipio", "tributo", "modelo", "origin", "step"]
    ref = pd.read_csv(cfg.forecasts_dir / "cv_all.csv", parse_dates=["origin"])
    new = pd.read_csv(out / "cv_all_rebuild.csv", parse_dates=["origin"])
    print(f"cache: {len(ref)} linhas | rebuild: {len(new)} linhas")
    j = ref.merge(new, on=keys, suffixes=("_ref", "_new"))
    print(f"linhas pareadas: {len(j)} "
          f"({100 * len(j) / max(len(ref), len(new)):.2f}%)")
    fails = 0
    if len(j) != len(ref) or len(ref) != len(new):
        print("FALHA: contagem de linhas nao bate")
        fails += 1
    for col in ("y_true", "y_pred", "insample_scale"):
        a = j[f"{col}_ref"].to_numpy()
        b = j[f"{col}_new"].to_numpy()
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-9)
        print(f"  {col:15s} desvio relativo max = {rel.max():.3e} "
              f"(mediano {np.median(rel):.3e})")
        if rel.max() > RTOL:
            worst = j.loc[np.argmax(rel), keys].to_dict()
            print(f"    FALHA (> {RTOL:.0e}); pior caso: {worst}")
            fails += 1
    refv = pd.read_csv(cfg.forecasts_dir / "sarima_var.csv",
                       parse_dates=["origin"])
    newv = pd.read_csv(out / "sarima_var_rebuild.csv", parse_dates=["origin"])
    kv = ["municipio", "tributo", "origin", "step"]
    jv = refv.merge(newv, on=kv, suffixes=("_ref", "_new"))
    for col in ("y_pred_mean",):
        rel = (np.abs(jv[f"{col}_ref"] - jv[f"{col}_new"])
               / np.maximum(np.abs(jv[f"{col}_ref"]), 1e-9)).to_numpy()
        print(f"  sarima_var {col}: pareadas {len(jv)}/{len(refv)}; "
              f"desvio relativo max = {rel.max():.3e}")
        if len(jv) != len(refv) or rel.max() > RTOL:
            print(f"    FALHA (> {RTOL:.0e} ou pareamento incompleto)")
            fails += 1
    if fails == 0:
        print("VERIFY OK: rebuild equivalente ao cache canonico.")
    return fails


def promote(cfg) -> None:
    out = _rebuild_dir(cfg)
    import shutil
    shutil.copy(out / "cv_all_rebuild.csv", cfg.forecasts_dir / "cv_all.csv")
    shutil.copy(out / "sarima_var_rebuild.csv",
                cfg.forecasts_dir / "sarima_var.csv")
    print("PROMOVIDO: cv_all.csv e sarima_var.csv substituidos pelo rebuild.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="append", required=False,
                    choices=["sf", "classic", "merge", "verify"],
                    help="Estagio(s) a executar, na ordem dada.")
    ap.add_argument("--promote", action="store_true",
                    help="Apos verify OK, substitui o cache canonico.")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    if not args.stage and not args.promote:
        ap.print_help()
        return 1
    cfg = load_config(args.config)
    rc = 0
    for st in args.stage or []:
        print(f"===== stage {st} =====", flush=True)
        if st == "sf":
            stage_sf(cfg)
        elif st == "classic":
            stage_classic(cfg)
        elif st == "merge":
            stage_merge(cfg)
        elif st == "verify":
            rc = stage_verify(cfg)
    if args.promote:
        if rc != 0:
            print("NAO promovido: verify falhou.")
            return rc
        promote(cfg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
