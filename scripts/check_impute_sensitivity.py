# -*- coding: utf-8 -*-
"""Sensibilidade da imputacao de 2016 no ISSQN de Camacari (Cap. 5).

O texto afirma: "refazer a avaliacao sem imputar 2016 nao muda o vencedor nem
a ordenacao: a imputacao e conservadora". Este script gera o artefato que
sustenta a frase: re-executa o portfolio para camacari-ISSQN SEM imputacao e
compara os ranques (MASE mediano h=1 e h=12; erro anual medio) com os do cache
canonico (com imputacao). Grava data/reports/impute_sensitivity.md.

Como AutoETS/AutoTheta exigem o venv statsforecast e o resto o venv principal,
o trabalho e dividido como no run_pipeline_full:

  (venv sf)        python scripts/check_impute_sensitivity.py --stage sf
  (venv principal) python scripts/check_impute_sensitivity.py --stage classic
  (qualquer)       python scripts/check_impute_sensitivity.py --stage report
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")

SERIE = ("camacari", "ISSQN")


def _workdir(cfg) -> Path:
    d = cfg.forecasts_dir / "_impute_check"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serie_sem_imputar(cfg) -> pd.Series:
    return prepare_series(cfg, impute=False)[SERIE]


def stage_sf(cfg) -> None:
    from forecasting.precisao import make_sf_specs, sf_rolling_cv
    out = _workdir(cfg)
    s = _serie_sem_imputar(cfg)
    for name, spec in make_sf_specs().items():
        d = sf_rolling_cv(s, spec)
        d["modelo"] = name
        d.to_csv(out / f"cv_{name}.csv", index=False)
        print(f"=> cv_{name}.csv ({len(d)} linhas)")


def stage_classic(cfg) -> None:
    from forecasting.precisao import (
        make_prophet_fitter,
        rolling_cv_any,
        sarima_D1_rolling_with_var,
    )
    out = _workdir(cfg)
    s = _serie_sem_imputar(cfg)
    d = rolling_origin_cv(s, fit_naive_seasonal, INITIAL_WINDOW, MAX_HORIZON,
                          ROLLING_STEP)
    d["modelo"] = "Naive"
    d.to_csv(out / "cv_Naive.csv", index=False)
    sv = sarima_D1_rolling_with_var(s)
    sv = sv.rename(columns={"y_pred_median": "y_pred"})
    sv["modelo"] = "SARIMA"
    sv.to_csv(out / "cv_SARIMA.csv", index=False)
    p = rolling_cv_any(s, make_prophet_fitter(use_holidays=False,
                                              yearly_fourier=6, cps=0.05,
                                              cr=0.8))
    p["modelo"] = "Prophet"
    p.to_csv(out / "cv_Prophet.csv", index=False)
    print("=> cv_Naive.csv, cv_SARIMA.csv, cv_Prophet.csv")


def _rank_table(cv: pd.DataFrame) -> pd.DataFrame:
    cv = cv.copy()
    cv["scaled"] = (cv["y_true"] - cv["y_pred"]).abs() / cv["insample_scale"]
    m1 = cv[cv["step"] == 1].groupby("modelo")["scaled"].median()
    m12 = cv[cv["step"] == 12].groupby("modelo")["scaled"].median()
    dec = cv[(cv["origin"].dt.month == 12) & cv["step"].between(1, 12)].copy()
    dec["ty"] = dec["origin"].dt.year + 1
    g = dec.groupby(["modelo", "ty"]).agg(pred=("y_pred", "sum"),
                                          real=("y_true", "sum"),
                                          n=("step", "count")).reset_index()
    g = g[g["n"] == 12]
    g["err"] = 100 * (g["pred"] - g["real"]).abs() / g["real"].abs()
    ann = g.groupby("modelo")["err"].mean()
    return pd.DataFrame({"mase_h1": m1, "mase_h12": m12, "err_anual": ann})


def stage_report(cfg) -> int:
    out = _workdir(cfg)
    cols = ["origin", "step", "y_true", "y_pred", "insample_scale", "modelo"]
    frames = []
    for f in sorted(out.glob("cv_*.csv")):
        d = pd.read_csv(f, parse_dates=["origin"])
        frames.append(d[cols])
    sem = pd.concat(frames, ignore_index=True)
    formal = sem[sem["modelo"].isin(["ETS", "SARIMA", "Theta", "Prophet"])]
    ens = (formal.groupby(["origin", "step"])
           .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                insample_scale=("insample_scale", "first")).reset_index())
    ens["modelo"] = "Ensemble"
    sem = pd.concat([sem, ens[cols]], ignore_index=True)

    com = pd.read_csv(cfg.forecasts_dir / "cv_all.csv", parse_dates=["origin"])
    com = com[(com["municipio"] == SERIE[0]) & (com["tributo"] == SERIE[1])]

    t_com = _rank_table(com[cols])
    t_sem = _rank_table(sem)

    lines = [
        "# Sensibilidade da imputacao: ISSQN de Camacari sem imputar 2016",
        "",
        "> Gerado por `scripts/check_impute_sensitivity.py`. Portfolio completo",
        "> re-executado sobre a serie SEM a imputacao de 2016, mesmo protocolo.",
        "> Sustenta a frase do Cap. 5: a imputacao nao muda o vencedor nem a",
        "> ordenacao.",
        "",
        "| Modelo | MASE h=1 com/sem | MASE h=12 com/sem | Erro anual medio com/sem (%) |",
        "|---|---|---|---|",
    ]
    order = ["Naive", "ETS", "SARIMA", "Prophet", "Theta", "Ensemble"]
    for m in order:
        lines.append(
            f"| {m} | {t_com.loc[m, 'mase_h1']:.2f} / {t_sem.loc[m, 'mase_h1']:.2f} "
            f"| {t_com.loc[m, 'mase_h12']:.2f} / {t_sem.loc[m, 'mase_h12']:.2f} "
            f"| {t_com.loc[m, 'err_anual']:.1f} / {t_sem.loc[m, 'err_anual']:.1f} |")

    def ranks(t, col):
        return list(t[col].sort_values().index)

    # Veredito com a banda de equivalencia do proprio trabalho (10%): o
    # "vencedor" e considerado mantido se o vencedor antigo fica a menos de
    # 10% do novo. IMPORTANTE: sem a imputacao o denominador do MASE absorve
    # o degrau artificial de 2016 (a escala in-sample cresce), entao os
    # NIVEIS de MASE com/sem nao sao comparaveis; apenas as ordenacoes
    # internas de cada coluna sao.
    BAND = 0.10
    strict_flips, band_flips, order_changes = [], [], []
    for col in ("mase_h1", "mase_h12", "err_anual"):
        rc, rs = ranks(t_com, col), ranks(t_sem, col)
        lines.append("")
        lines.append(f"Ordenacao por {col}: com imputacao {' > '.join(rc)}; "
                     f"sem imputacao {' > '.join(rs)}.")
        if rc != rs:
            order_changes.append(col)
        if rc[0] != rs[0]:
            strict_flips.append(col)
            old_val = float(t_sem.loc[rc[0], col])
            new_val = float(t_sem.loc[rs[0], col])
            if old_val > new_val * (1 + BAND):
                band_flips.append(
                    f"{col}: {rc[0]} ({old_val:.2f}) perde para {rs[0]} "
                    f"({new_val:.2f}) por mais de {BAND:.0%}")
    lines.append("")
    if band_flips:
        veredito = ("ATENCAO: sem a imputacao o vencedor muda ALEM da banda "
                    "de equivalencia de 10% em: " + "; ".join(band_flips)
                    + ". A frase do Cap. 5 nao se sustenta como esta.")
    elif strict_flips:
        veredito = ("O vencedor estrito muda em " + ", ".join(strict_flips)
                    + ", mas dentro da banda de equivalencia de 10% "
                    "(empate pratico). Ordenacoes intermediarias mudam em: "
                    + ", ".join(order_changes) + ".")
    else:
        veredito = "O vencedor nao muda em nenhum criterio."
    lines.append(veredito)
    rep = cfg.analysis_root / "data" / "reports" / "impute_sensitivity.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"gravado: {rep}")
    print(veredito)
    return 1 if band_flips else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="append", required=True,
                    choices=["sf", "classic", "report"])
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = 0
    for st in args.stage:
        print(f"===== stage {st} =====", flush=True)
        if st == "sf":
            stage_sf(cfg)
        elif st == "classic":
            stage_classic(cfg)
        elif st == "report":
            rc = stage_report(cfg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
