# -*- coding: utf-8 -*-
"""Verificacao independente dos resultados centrais do TCC.

Nao importa NADA do pacote ``forecasting``: le apenas os artefatos CSV crus e
recomputa, do zero, os numeros que o texto afirma. Escrito para responder, com
evidencia, a pergunta da banca: "se os modelos nao vencem o baseline em h=12,
como vencem a prefeitura?".

Recomputa:
  A. Sanidade do cache (modelos, series, dobras por horizonte).
  B. MASE por modelo em h=1 e h=12 (mediana e media), contra a Tabela 2.
  C. Agregacao anual (origens de dezembro, 12 passos, anos completos, correcao
     de Jensen no SARIMA) e o placar anual contra a prefeitura, por modelo --
     contra a Tabela 4 (24/30 do Ensemble etc.).
  D. A resposta ao paradoxo: erro anual tipico da prefeitura vs Naive vs
     Ensemble; placar do PROPRIO Naive contra a prefeitura; distancia
     Ensemble-Naive em MASE h=12 frente a banda de equivalencia de 10%.
  E. Robustez de base: converte as previsoes dos modelos de volta a NOMINAL
     (pelo indice IPCA) e refaz o placar -- o 24/30 nao pode depender da base.

Uso:  python analysis/scripts/verificacao_independente.py
Saida: relatorio no stdout (redirecione para arquivo se quiser versionar).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CV = REPO / "analysis" / "data" / "forecasts" / "cv_all.csv"
SARIMA_VAR = REPO / "analysis" / "data" / "forecasts" / "sarima_var.csv"
PREF = REPO / "siconfi-collector" / "data" / "transformed" / "prefeitura_forecast.csv"
IPCA = REPO / "analysis" / "data" / "ipca_sgs433.csv"
MONTHLY = REPO / "siconfi-collector" / "data" / "transformed" / "monthly_revenue.csv"

IBGE = {"salvador": 2927408, "camacari": 2905701, "ilheus": 2913606}
MODELS = ["Naive", "ETS", "SARIMA", "Prophet", "Theta", "Ensemble"]


def h(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    cv = pd.read_csv(CV, parse_dates=["origin", "target_date"])
    pref = pd.read_csv(PREF)
    ipca = pd.read_csv(IPCA)

    # ---------- A. Sanidade do cache -------------------------------------
    h("A. SANIDADE DO CACHE (cv_all.csv)")
    print(f"linhas: {len(cv)}")
    print(f"modelos: {sorted(cv['modelo'].unique())}")
    series = cv.groupby(["municipio", "tributo"]).size()
    print(f"series ({len(series)}): {list(series.index)}")
    print(f"origens: {cv['origin'].min():%Y-%m} a {cv['origin'].max():%Y-%m}")
    for step in (1, 12):
        n = len(cv[(cv["step"] == step) & (cv["modelo"] == "Naive")])
        print(f"dobras com step={step} (por modelo): {n}  "
              f"(Tabela 2 declara {360 if step == 1 else 294})")

    # Amarra o y_true ao dado nominal cru: y_true = nominal * I_base / I_t.
    mon = pd.read_csv(MONTHLY, parse_dates=["date"])
    idx = ipca.set_index(pd.PeriodIndex(pd.to_datetime(ipca["date"]), freq="M"))[
        "ipca_index"]
    base_p = pd.Period("2025-12", freq="M")
    sample = cv[(cv["municipio"] == "salvador") & (cv["tributo"] == "IPTU")
                & (cv["target_date"] == "2021-01-01")].iloc[0]
    nom = mon[(mon["cod_ibge"] == IBGE["salvador"])
              & (mon["date"] == "2021-01-01")]["iptu"].iloc[0]
    expected = nom * float(idx.loc[base_p]) / float(idx.loc[pd.Period("2021-01", "M")])
    print(f"amarracao nominal->real (Salvador IPTU 2021-01): "
          f"y_true={sample['y_true']:.2f}  nominal*deflator={expected:.2f}  "
          f"dif={abs(sample['y_true'] - expected):.4f}")

    # ---------- B. MASE em h=1 e h=12 ------------------------------------
    h("B. MASE POR MODELO (recomputado por dobra: |erro|/escala in-sample)")
    cv["scaled_err"] = (cv["y_true"] - cv["y_pred"]).abs() / cv["insample_scale"]
    for step in (1, 12):
        sub = cv[cv["step"] == step]
        stats = sub.groupby("modelo")["scaled_err"].agg(["median", "mean"])
        print(f"\n  step == {step} (isolado):")
        for m in MODELS:
            if m in stats.index:
                print(f"    {m:<9} mediana={stats.loc[m, 'median']:.2f}  "
                      f"media={stats.loc[m, 'mean']:.2f}")
    # Alternativa: MASE da trajetoria completa da dobra (media dos 12 passos).
    traj = (cv[cv["step"].between(1, 12)]
            .groupby(["municipio", "tributo", "modelo", "origin"])
            .agg(mase_traj=("scaled_err", "mean"), n=("step", "count"))
            .reset_index())
    traj = traj[traj["n"] == 12]
    stats = traj.groupby("modelo")["mase_traj"].agg(["median", "mean"])
    print("\n  trajetoria h=1..12 completa por dobra (media dos 12 passos):")
    for m in MODELS:
        if m in stats.index:
            print(f"    {m:<9} mediana={stats.loc[m, 'median']:.2f}  "
                  f"media={stats.loc[m, 'mean']:.2f}")
    print("  -> comparar com a Tabela 2 para identificar qual definicao o texto usa.")

    # ---------- C. Agregacao anual + placar vs prefeitura -----------------
    h("C. PLACAR ANUAL CONTRA A PREFEITURA (base real, como no pipeline)")
    dec = cv[(cv["origin"].dt.month == 12) & (cv["step"].between(1, 12))].copy()
    dec["target_year"] = dec["origin"].dt.year + 1
    annual = (dec.groupby(["municipio", "tributo", "modelo", "target_year"])
              .agg(pred=("y_pred", "sum"), real=("y_true", "sum"),
                   n=("step", "count")).reset_index())
    annual = annual[annual["n"] == 12]

    # Correcao de Jensen no agregado anual do SARIMA (se o cache existir).
    if SARIMA_VAR.exists():
        var = pd.read_csv(SARIMA_VAR, parse_dates=["origin"])
        jdec = var[(var["origin"].dt.month == 12) & (var["step"].between(1, 12))].copy()
        jdec["target_year"] = jdec["origin"].dt.year + 1
        jen = (jdec.groupby(["municipio", "tributo", "target_year"])
               .agg(pred_jen=("y_pred_mean", "sum"), n=("step", "count"))
               .reset_index())
        jen = jen[jen["n"] == 12].drop(columns="n")
        annual = annual.merge(jen, on=["municipio", "tributo", "target_year"],
                              how="left")
        is_sar = (annual["modelo"] == "SARIMA") & annual["pred_jen"].notna()
        annual.loc[is_sar, "pred"] = annual.loc[is_sar, "pred_jen"]
        annual = annual.drop(columns="pred_jen")
        print(f"correcao de Jensen aplicada ao SARIMA em {int(is_sar.sum())} "
              "anos-serie (cache sarima_var.csv presente)")
    else:
        print("AVISO: sarima_var.csv ausente; agregado do SARIMA sem Jensen.")

    annual["err_model"] = 100.0 * (annual["pred"] - annual["real"]).abs() / annual["real"].abs()

    code2key = {v: k for k, v in IBGE.items()}
    pf = pref[pref["cod_ibge"].isin(code2key)].copy()
    pf["municipio"] = pf["cod_ibge"].map(code2key)
    pf = pf[["municipio", "tributo", "year", "erro_pct_prefeitura", "periodo_fonte"]]
    pf = pf.rename(columns={"year": "target_year"})
    print(f"fonte da prefeitura: periodo_fonte por linha = "
          f"{sorted(pf['periodo_fonte'].unique())} (1 = P1, proximo da LOA)")

    m = annual.merge(pf, on=["municipio", "tributo", "target_year"], how="inner")
    m["beat"] = m["err_model"] < m["erro_pct_prefeitura"]

    anos = sorted(m["target_year"].unique())
    n_ys = m.groupby("modelo")["beat"].count()
    print(f"anos-teste: {anos}")
    print(f"anos-serie por modelo: {n_ys.iloc[0]}")
    print("\n  placar (venceu a prefeitura) e erro anual medio por modelo:")
    for mod in MODELS:
        b = m[m["modelo"] == mod]
        print(f"    {mod:<9} {int(b['beat'].sum()):>2}/{len(b)}  "
              f"({100 * b['beat'].mean():.0f}%)   erro medio {b['err_model'].mean():.1f}%  "
              f"mediano {b['err_model'].median():.1f}%")
    pref_series = m.groupby(["municipio", "tributo", "target_year"])[
        "erro_pct_prefeitura"].first()
    print(f"\n  erro da prefeitura: medio {pref_series.mean():.1f}%  "
          f"mediano {pref_series.median():.1f}%")

    # Melhor ex-post por serie (criterio do texto: menor erro anual MEDIO).
    best = (m.groupby(["municipio", "tributo", "modelo"])["err_model"].mean()
            .reset_index()
            .sort_values("err_model")
            .groupby(["municipio", "tributo"]).first().reset_index())
    tot_b, tot_n = 0, 0
    print("\n  melhor modelo ex-post por serie (menor erro anual medio):")
    for _, r in best.iterrows():
        b = m[(m["municipio"] == r["municipio"]) & (m["tributo"] == r["tributo"])
              & (m["modelo"] == r["modelo"])]
        tot_b += int(b["beat"].sum())
        tot_n += len(b)
        print(f"    {r['municipio']:<9} {r['tributo']:<5} -> {r['modelo']:<9} "
              f"venceu {int(b['beat'].sum())}/{len(b)}  erro medio {r['err_model']:.1f}%")
    print(f"    TOTAL ex-post: {tot_b}/{tot_n} ({100 * tot_b / tot_n:.0f}%)")

    # ---------- D. O paradoxo, quantificado -------------------------------
    h("D. O PARADOXO: perder do Naive em h=12 e vencer a prefeitura")
    naive = m[m["modelo"] == "Naive"]
    ens = m[m["modelo"] == "Ensemble"]
    print(f"  Naive   vs prefeitura: {int(naive['beat'].sum())}/{len(naive)} "
          f"({100 * naive['beat'].mean():.0f}%)")
    print(f"  Ensemble vs prefeitura: {int(ens['beat'].sum())}/{len(ens)} "
          f"({100 * ens['beat'].mean():.0f}%)")
    # Confronto direto anual Ensemble vs Naive (mesmos anos-serie):
    both = ens.merge(naive, on=["municipio", "tributo", "target_year"],
                     suffixes=("_ens", "_nv"))
    ens_win = (both["err_model_ens"] < both["err_model_nv"]).sum()
    print(f"  Ensemble vs Naive (anual, direto): Ensemble melhor em "
          f"{int(ens_win)}/{len(both)} anos-serie")
    med = cv[cv["step"] == 12].groupby("modelo")["scaled_err"].median()
    gap = 100 * (med["Ensemble"] - med["Naive"]) / med["Naive"]
    print(f"  MASE mediano h=12: Naive {med['Naive']:.2f} vs Ensemble "
          f"{med['Ensemble']:.2f}  (gap {gap:+.1f}%; banda de equivalencia do "
          "texto: 10%)")
    ratio = pref_series.mean() / m[m['modelo'] == 'Ensemble']['err_model'].mean()
    print(f"  erro anual medio: prefeitura {pref_series.mean():.1f}% vs Ensemble "
          f"{ens['err_model'].mean():.1f}%  (razao {ratio:.1f}x)")

    # ---------- E. Robustez: placar em base NOMINAL -----------------------
    h("E. ROBUSTEZ DE BASE: placar refeito em NOMINAL (modelos re-inflacionados)")
    per = pd.PeriodIndex(dec["target_date"], freq="M")
    factor = idx.reindex(per).to_numpy() / float(idx.loc[base_p])
    dec_nom = dec.copy()
    dec_nom["y_pred_nom"] = dec_nom["y_pred"].to_numpy() * factor
    dec_nom["y_true_nom"] = dec_nom["y_true"].to_numpy() * factor
    an_nom = (dec_nom.groupby(["municipio", "tributo", "modelo", "target_year"])
              .agg(pred=("y_pred_nom", "sum"), real=("y_true_nom", "sum"),
                   n=("step", "count")).reset_index())
    an_nom = an_nom[an_nom["n"] == 12]
    an_nom["err_model"] = 100.0 * (an_nom["pred"] - an_nom["real"]).abs() / an_nom["real"].abs()
    mn = an_nom.merge(pf, on=["municipio", "tributo", "target_year"], how="inner")
    mn["beat"] = mn["err_model"] < mn["erro_pct_prefeitura"]
    print("  (sem Jensen no SARIMA nesta variante; o efeito e pequeno e so no SARIMA)")
    for mod in MODELS:
        b = mn[mn["modelo"] == mod]
        print(f"    {mod:<9} {int(b['beat'].sum()):>2}/{len(b)}  "
              f"({100 * b['beat'].mean():.0f}%)   erro medio {b['err_model'].mean():.1f}%")
    print("  -> se o placar nominal divergir muito do real (secao C), a base "
          "da comparacao e fragil; se nao, e robusta.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
