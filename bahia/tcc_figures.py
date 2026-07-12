# -*- coding: utf-8 -*-
"""Versoes das figuras da analise estadual NO PADRAO VISUAL DO TCC.

Reaproveitam os dados de out/ (cv_bahia.csv, best_model.json, malha) e o
cache canonico (cv_all.csv) e aplicam a casa de estilo da tese
(setup_matplotlib_thesis: fontes, virgula BR, fundo transparente, sem titulo
de matplotlib -- o titulo vem do \\caption). Salvam PDF em tcc-latex/figures/
generated/ com os nomes fig_bahia_*.pdf e fig_intervalo_modelos.pdf.

Gera:
  fig_bahia_mapa           coropletico da vantagem do Ensemble por municipio
  fig_bahia_margem         margem mediana e taxa de vitoria por faixa de porte
  fig_bahia_melhor_modelo  Ensemble fixo vs. melhor modelo (ex-post e oraculo)
  fig_intervalo_modelos    faixa de previsao do portfolio (dispersao entre
                           modelos) alargando com o horizonte -- uso Munitax
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS / "src"))
OUT = HERE / "out"

import geodata  # noqa: E402
from forecasting.config import (MODEL_COLORS, MUNITAX_BLUE,  # noqa: E402
                                load_config)
from forecasting.plotting import (BAR_COMPARISON, BAR_LABEL_SIZE,  # noqa: E402
                                  REALIZED_INK, br_axis, pixel_scale,
                                  rounded_bar, save_figure,
                                  setup_matplotlib_thesis, style_axis)

INK, MUTED, BORDER = "#202124", "#5F6368", "#DADCE0"
CFG = load_config()
FIGDIR = CFG.figures_dir_abs


# ---------- dados compartilhados -------------------------------------------


def _confrontos() -> pd.DataFrame:
    from forecasting.io import load_prefeitura_forecast
    cv = pd.read_csv(OUT / "cv_bahia.csv")
    cv["err"] = (100 * (cv["pred_annual"] - cv["real_annual"]).abs()
                 / cv["real_annual"].abs())
    pf = load_prefeitura_forecast(CFG).rename(columns={"year": "target_year"})
    m = cv.merge(pf[["cod_ibge", "tributo", "target_year",
                     "erro_pct_prefeitura"]],
                 on=["cod_ibge", "tributo", "target_year"], how="inner")
    m["vence"] = m["err"] < m["erro_pct_prefeitura"]
    return m


# ---------- 1. mapa ---------------------------------------------------------


def mapa() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    setup_matplotlib_thesis()
    m = _confrontos()
    ens = m[m["modelo"] == "Ensemble"].copy()
    ens["vant"] = ens["erro_pct_prefeitura"] - ens["err"]
    adv = ens.groupby("cod_ibge")["vant"].mean()

    polys = geodata.feature_polygons(geodata.load_malha())
    lim = max(10.0, round(float(np.nanpercentile(adv.abs(), 80)) / 5) * 5)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    cmap = LinearSegmentedColormap.from_list(
        "vant", ["#B3564A", "#E5B9B2", "#FFFFFF", "#9CC8F7", MUNITAX_BLUE])

    fig, ax = plt.subplots(figsize=(6.0, 6.3))
    com, sem, cores = [], [], []
    for cod, rings in polys.items():
        for ring in rings:
            xy = np.asarray(ring)
            if cod in adv.index:
                com.append(xy)
                cores.append(cmap(norm(np.clip(adv[cod], -lim, lim))))
            else:
                sem.append(xy)
    # Sem dados: cinza neutro SOLIDO, inequivocamente distinto do branco (=zero)
    # da rampa divergente. (A hachura anterior usava a cor do edge, branca, e
    # ficava invisivel.)
    ax.add_collection(PolyCollection(sem, facecolors="#D9DDE2",
                                     edgecolors="white", linewidths=0.25))
    ax.add_collection(PolyCollection(com, facecolors=cores,
                                     edgecolors="white", linewidths=0.25))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.01)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04,
                        pad=0.0, aspect=32, extend="both")
    cbar.set_label("vantagem do Ensemble sobre a projeção oficial "
                   "(pontos percentuais de erro anual)", fontsize=8.0,
                   color=MUTED)
    cbar.ax.tick_params(labelsize=7.6, colors=MUTED)
    cbar.ax.xaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(
            lambda x, _p: f"{x:.0f}".replace("-", "−")))
    cbar.outline.set_visible(False)
    save_figure(fig, "fig_bahia_mapa", FIGDIR)
    plt.close(fig)
    print("+ fig_bahia_mapa")


# ---------- 2. margem por faixa --------------------------------------------


FAIXAS = [(0, 20_000, "até 20 mil"), (20_000, 50_000, "20 a 50 mil"),
          (50_000, 100_000, "50 a 100 mil"), (100_000, None, "acima de 100 mil")]


def margem() -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib_thesis()
    m = _confrontos()
    ens = m[m["modelo"] == "Ensemble"].merge(
        geodata.load_population()[["cod_ibge", "populacao"]], on="cod_ibge",
        how="left").dropna(subset=["populacao"])
    ens["vant"] = ens["erro_pct_prefeitura"] - ens["err"]
    muns = ens.groupby("cod_ibge").agg(
        populacao=("populacao", "first"), vant=("vant", "mean")).reset_index()

    rows = []
    for lo, hi, rot in FAIXAS:
        sel = ens[(ens["populacao"] >= lo)
                  & ((hi is None) | (ens["populacao"] < (hi or np.inf)))]
        selm = muns[(muns["populacao"] >= lo)
                    & ((hi is None) | (muns["populacao"] < (hi or np.inf)))]
        rows.append((rot, float(selm["vant"].median()),
                     100 * sel["vence"].mean()))

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    xs = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, max(vals) * 1.2)
    bw = 0.58
    scale = pixel_scale(ax)
    for x, (rot, mg, wr) in zip(xs, rows):
        rounded_bar(ax, x - bw / 2, 0, bw, mg, MUNITAX_BLUE, scale=scale)
        ax.text(x, mg + 0.35, f"{mg:.1f}".replace(".", ",") + " p.p.",
                ha="center", va="bottom", fontsize=8.5, fontweight="semibold",
                color=INK)
        ax.text(x, 0.5, f"vence {wr:.0f}%".replace(".", ","), ha="center",
                fontsize=7.2, color="white", zorder=3)
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.set_ylabel("Margem mediana de vantagem (p.p.)")
    br_axis(ax, "y", decimals=0, step=4)
    style_axis(ax)
    ax.tick_params(axis="x", length=0)
    save_figure(fig, "fig_bahia_margem", FIGDIR)
    plt.close(fig)
    print("+ fig_bahia_margem")


# ---------- 3. Ensemble fixo vs. melhor modelo -----------------------------


def melhor_modelo() -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib_thesis()
    bm = json.loads((OUT / "best_model.json").read_text(encoding="utf-8"))
    fx, es = bm["modelos_fixos"], bm["estrategias"]
    oraculo = bm.get("oraculo_ano", {}).get("taxa")

    linhas = []
    if oraculo is not None:
        linhas.append(("Oráculo: melhor modelo a cada ano", oraculo, "teto"))
    linhas += [
        ("Melhor por série, ex-post (à la Oliveira)",
         es["melhor_expost"]["taxa"], "teto"),
        ("Ensemble fixo (adotado)", fx["Ensemble"]["taxa"], "adotado"),
        ("Seleção por série sem ver o futuro",
         es["melhor_walkforward"]["taxa"], "deploy"),
    ]
    cor = {"teto": "#EDEEF0", "adotado": MUNITAX_BLUE, "deploy": "#B7BCC2"}
    h = 0.46
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ys = np.arange(len(linhas))[::-1]
    ax.set_xlim(0, 92)
    ax.set_ylim(-0.62, len(linhas) - 0.30)
    scale = pixel_scale(ax)
    for y, (rot, tx, tipo) in zip(ys, linhas):
        # rotulo ACIMA da barra, alinhado a x=0: sem coluna de rotulos a
        # direita, sem vao entre o texto e o inicio da barra.
        ax.text(0, y + h / 2 + 0.12, rot, ha="left", va="bottom",
                fontsize=8.2, color=INK)
        kw = dict(hatch="////", edgecolor="#9AA0A6") if tipo == "teto" else {}
        rounded_bar(ax, 0, y - h / 2, tx, h, cor[tipo], scale=scale, **kw)
        import matplotlib.patheffects as pe
        ax.text(tx + 1.1, y, f"{tx:.1f}%".replace(".", ","), va="center",
                fontsize=8.5, fontweight="semibold", color=INK, zorder=4,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.axvline(fx["Ensemble"]["taxa"], color=MUTED, lw=0.9, ls=(0, (4, 3)),
               zorder=1)
    ax.set_yticks([])
    ax.set_xlabel("Anos-série em que vence a projeção da prefeitura (%)")
    ax.xaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(lambda x, _p: f"{x:.0f}"))
    style_axis(ax)
    ax.tick_params(axis="y", length=0)
    ax.grid(False, axis="y")
    save_figure(fig, "fig_bahia_melhor_modelo", FIGDIR)
    plt.close(fig)
    print("+ fig_bahia_melhor_modelo")


# ---------- 4. faixa de previsao do portfolio (Munitax) --------------------


def intervalo_modelos() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    setup_matplotlib_thesis()
    cv = pd.read_csv(ANALYSIS / "data/forecasts/cv_all.csv",
                     parse_dates=["origin", "target_date"])
    serie_mun, serie_trib = "salvador", "ISSQN"  # cv 0,16: tendencia limpa
    origem = pd.Timestamp("2024-12-01")  # ano em que o Ensemble melhor segue
    formais = ["ETS", "SARIMA", "Theta", "Prophet"]

    sub = cv[(cv["municipio"] == serie_mun) & (cv["tributo"] == serie_trib)]
    # serie realizada (deflacionada) reconstruida do cache
    real = (sub.drop_duplicates("target_date")
            .set_index("target_date")["y_true"].sort_index())
    janela = real[(real.index >= origem - pd.DateOffset(months=24))
                  & (real.index <= origem + pd.DateOffset(months=12))]

    fc = sub[sub["origin"] == origem]
    piv = fc.pivot_table(index="target_date", columns="modelo",
                         values="y_pred").sort_index()
    lo, hi, ens = piv[formais].min(axis=1), piv[formais].max(axis=1), \
        piv["Ensemble"]
    # ancora a banda no ultimo ponto observado (Timestamp puro em pandas: sem
    # mistura de tipos que embaralha o eixo de datas)
    anc = pd.Series([float(real.loc[origem])], index=[origem])
    lo, hi, ens = (pd.concat([anc, s]).sort_index() for s in (lo, hi, ens))

    esc = 1e6  # eixo em R$ milhoes
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.fill_between(lo.index, lo / esc, hi / esc, color=MUNITAX_BLUE,
                    alpha=0.16, lw=0, zorder=1, label="Faixa entre os modelos")
    ax.plot(lo.index, lo / esc, color=MUNITAX_BLUE, lw=0.8, alpha=0.5, zorder=2)
    ax.plot(hi.index, hi / esc, color=MUNITAX_BLUE, lw=0.8, alpha=0.5, zorder=2)
    ax.plot(ens.index, ens / esc, color=MODEL_COLORS["Ensemble"], lw=1.8,
            zorder=4, label="Ensemble")
    ax.plot(janela.index, janela.to_numpy() / esc, color=REALIZED_INK,
            lw=1.6, zorder=5, label="Realizado")
    ax.axvline(origem, color=BORDER, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.annotate("origem da previsão", xy=(origem, 0.02),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=7.0, color=MUTED,
                va="bottom", ha="left")

    ax.set_ylabel("Arrecadação mensal (R$ milhões)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax.margins(x=0.02)
    br_axis(ax, "y", decimals=0)
    style_axis(ax)
    from forecasting.plotting import clean_legend
    # ordem canonica, igual a fig_forecasts_formais: realizado, faixa, ensemble
    h, l = _h_l(ax)
    ordem = ["Realizado", "Faixa entre os modelos", "Ensemble"]
    hl = dict(zip(l, h))
    clean_legend(fig, [hl[k] for k in ordem], ordem, ncol=3)
    save_figure(fig, "fig_intervalo_modelos", FIGDIR)
    plt.close(fig)
    print("+ fig_intervalo_modelos")


def _h_l(ax):
    h, l = ax.get_legend_handles_labels()
    return list(h), list(l)


def main() -> None:
    mapa()
    margem()
    melhor_modelo()
    intervalo_modelos()
    print(f"-> {FIGDIR}")


if __name__ == "__main__":
    main()
