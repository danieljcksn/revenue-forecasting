# -*- coding: utf-8 -*-
"""Recomputa e ASSEVERA os numeros canonicos citados na prosa do TCC.

Um unico comando responde "de onde vem esse numero?" para os quatro niveis:

  1. NUCLEO (Secao 5.6): Ensemble fixo 24/30 (80%); melhor ex-post 22/30 (73%).
  2. HIERARQUIA TEMPORAL (Secao 5.5): melhor mensal-agregado 9,0% vs melhor
     anual-direto 10,0% (media do painel B da figura), ARIMA mensal 11,0% vs
     anual 15,7% (painel A). Tambem grava data/reports/hierarquia_temporal.md.
  3. AMPLIADO (Secao 5.7): 155 confrontos; ex-post 70%; Ensemble fixo 63%;
     Ensemble com erro medio menor que a prefeitura em 24 das 31 series.
  4. BAHIA (Secao 5.8): Ensemble 67,0%; ex-post 74,2%; 188/230 municipios
     (le analysis/bahia/out/resumo.json, gerado por bahia/best_model.py).

Sai com codigo != 0 se qualquer numero canonico nao bater (tolerancia: o
arredondamento com que o numero aparece no texto).

Uso:  python scripts/check_winrates.py          (venv principal; sem statsforecast)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.benchmarks import _prefeitura_errors, aggregate_monthly_to_annual
from forecasting.config import load_config, series_keys
from forecasting.eda import prepare_series
from forecasting.evaluation import load_cv

FAILS: list[str] = []


def check(label: str, got, expected, fmt="{}"):
    ok = got == expected
    mark = "ok" if ok else "DIVERGE"
    print(f"  [{mark}] {label}: {fmt.format(got)} (esperado {fmt.format(expected)})")
    if not ok:
        FAILS.append(label)


def main() -> int:
    cfg = load_config()

    # ---------- 1. Nucleo -------------------------------------------------
    print("== 1. Nucleo (Secao 5.6): confronto anual com a prefeitura ==")
    annual = aggregate_monthly_to_annual(load_cv(cfg), cfg)
    pref = _prefeitura_errors(cfg)
    m = annual.merge(pref, on=["municipio", "tributo", "target_year"], how="inner")
    m["beat"] = m["err_pct_model"] < m["erro_pct_prefeitura"]
    ens = m[m["modelo"] == "Ensemble"]
    expost = 0
    for (_mk, _tr), b in m.groupby(["municipio", "tributo"]):
        best = b.groupby("modelo")["err_pct_model"].mean().idxmin()
        expost += int(b[b["modelo"] == best]["beat"].sum())
    check("Ensemble fixo (anos-serie)", f"{int(ens['beat'].sum())}/{len(ens)}", "24/30")
    check("Ensemble fixo (%)", round(100 * ens["beat"].mean()), 80)
    check("melhor ex-post (anos-serie)", f"{expost}/{len(ens)}", "22/30")
    check("melhor ex-post (%)", round(100 * expost / len(ens)), 73)

    # ---------- 2. Hierarquia temporal ------------------------------------
    # Fonte unica do experimento em forecasting.precisao (mesma usada pelo
    # gerador da figura, scripts/fig_hierarquia_temporal.py).
    print("== 2. Hierarquia temporal (Secao 5.5): mensal-agregada vs anual-direta ==")
    from forecasting.precisao import hierarquia_anual, hierarquia_mensal
    mens = hierarquia_mensal(cfg)
    ann = hierarquia_anual(cfg)

    m_by = mens.groupby("fam")["err"].mean()
    a_by = ann.groupby("fam")["err"].mean()
    mb = mens.groupby(["serie", "modelo"])["err"].mean().groupby("serie").min()
    ab = ann.groupby(["serie", "fam"])["err"].mean().groupby("serie").min()
    check("painel A: ARIMA mensal (%)", round(float(m_by["ARIMA"]), 1), 11.0,
          "{:.1f}")
    check("painel A: ARIMA anual (%)", round(float(a_by["ARIMA"]), 1), 15.7,
          "{:.1f}")
    check("painel B: media do melhor mensal (%)", round(float(mb.mean()), 1), 9.0,
          "{:.1f}")
    check("painel B: media do melhor anual (%)", round(float(ab.mean()), 1), 10.0,
          "{:.1f}")

    report = cfg.analysis_root / "data" / "reports" / "hierarquia_temporal.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Hierarquia temporal: proveniencia dos numeros da Secao 5.5",
        "",
        "> Gerado por `scripts/check_winrates.py`. Recorte identico ao da",
        "> figura fig_hierarquia_temporal: mensal = familias ARIMA (SARIMA com",
        "> correcao de Jensen), ETS e Naive do cache canonico; anual-direta =",
        "> rw/holt/holt amortecido/ARIMA(log) re-ajustados sobre os totais",
        "> anuais, 2021 a 2025, sem drift.",
        "",
        "## Painel A: erro anual medio por familia (%)",
        "",
        "| Familia | Mensal-agregada | Anual-direta |",
        "|---|---|---|",
    ]
    for fam in ("ARIMA", "ETS", "Naive/RW"):
        lines.append(f"| {fam} | {m_by.get(fam, float('nan')):.1f} | "
                     f"{a_by.get(fam, float('nan')):.1f} |")
    lines += [
        "",
        "## Painel B: melhor de cada via, por serie (%)",
        "",
        "| Serie | Melhor mensal | Melhor anual |",
        "|---|---|---|",
    ]
    for serie in sorted(mb.index):
        lines.append(f"| {serie} | {mb[serie]:.1f} | {ab[serie]:.1f} |")
    lines += [
        "",
        f"**Media do painel B: mensal {mb.mean():.2f}% vs anual {ab.mean():.2f}%**",
        "(arredondadas para 9,0% e 10,0% na prosa da Secao 5.5).",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"  relatorio: {report}")

    # ---------- 3. Ampliado ------------------------------------------------
    print("== 3. Ampliado (Secao 5.7): 31 series ==")
    from forecasting.generalization import _load_extended_cv
    from forecasting.io import load_prefeitura_forecast
    ext = _load_extended_cv(cfg).copy()
    ext["origin"] = pd.to_datetime(ext["origin"])
    dec = ext[(ext["origin"].dt.month == 12) & (ext["step"].between(1, 12))].copy()
    dec["target_year"] = dec["origin"].dt.year + 1
    ga = (dec.groupby(["cod_ibge", "tributo", "modelo", "target_year"])
          .agg(pred=("y_pred", "sum"), real=("y_true", "sum"), n=("step", "count"))
          .reset_index())
    ga = ga[ga["n"] == 12]
    ga["err"] = 100 * (ga["pred"] - ga["real"]).abs() / ga["real"].abs()
    pf = load_prefeitura_forecast(cfg).rename(columns={"year": "target_year"})
    ma = ga.merge(pf[["cod_ibge", "tributo", "target_year", "erro_pct_prefeitura"]],
                  on=["cod_ibge", "tributo", "target_year"], how="inner")
    ma["beat"] = ma["err"] < ma["erro_pct_prefeitura"]
    n_conf = ma[["cod_ibge", "tributo", "target_year"]].drop_duplicates().shape[0]
    expost_a = 0
    for (_c, _t), b in ma.groupby(["cod_ibge", "tributo"]):
        best = b.groupby("modelo")["err"].mean().idxmin()
        expost_a += int(b[b["modelo"] == best]["beat"].sum())
    ens_a = ma[ma["modelo"] == "Ensemble"]
    pts = ens_a.groupby(["cod_ibge", "tributo"]).agg(
        modelo_err=("err", "mean"), pref_err=("erro_pct_prefeitura", "mean"))
    check("confrontos", n_conf, 155)
    check("ex-post (%)", round(100 * expost_a / n_conf), 70)
    check("Ensemble fixo (%)", round(100 * ens_a["beat"].mean()), 63)
    check("series com Ensemble melhor",
          f"{int((pts['modelo_err'] < pts['pref_err']).sum())}/{len(pts)}", "24/31")

    # ---------- 4. Bahia ---------------------------------------------------
    print("== 4. Bahia (Secao 5.8): 230 municipios ==")
    resumo = json.loads(
        (cfg.analysis_root / "bahia" / "out" / "resumo.json").read_text("utf-8"))
    check("Ensemble win-rate (%)", resumo["ens_winrate"], 67.0, "{:.1f}")
    check("ex-post win-rate (%)", resumo["expost_winrate"], 74.2, "{:.1f}")
    check("municipios com vantagem",
          f"{resumo['municipios_vantagem_positiva']}/{resumo['municipios_total_mapa']}",
          "188/230")
    # Mediana do erro da LOA no estado ("erro anual mediano de 29%" na prosa):
    # recomputada do artefato da execucao estadual, orfao apontado pela
    # auditoria de proveniencia de 10/07.
    cvb = pd.read_csv(cfg.analysis_root / "bahia" / "out" / "cv_bahia.csv")
    pfb = load_prefeitura_forecast(cfg).rename(columns={"year": "target_year"})
    mb_ba = cvb[cvb["modelo"] == "Ensemble"].merge(
        pfb[["cod_ibge", "tributo", "target_year", "erro_pct_prefeitura"]],
        on=["cod_ibge", "tributo", "target_year"], how="inner")
    med_pref = float(mb_ba.groupby(["cod_ibge", "tributo", "target_year"])
                     ["erro_pct_prefeitura"].first().median())
    check("erro mediano da LOA no estado (%)", round(med_pref), 29)

    print()
    if FAILS:
        print(f"FALHOU: {len(FAILS)} numero(s) canonico(s) divergiram: {FAILS}")
        return 1
    print("OK: todos os numeros canonicos conferem com a prosa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
