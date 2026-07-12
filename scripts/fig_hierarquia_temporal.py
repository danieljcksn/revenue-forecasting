# -*- coding: utf-8 -*-
"""Gera fig_hierarquia_temporal.pdf (Secao 5.5): mensal-agregada vs anual-direta.

Consolidacao do gerador que vivia em _precisao_run/new_figures.py, agora
lendo o cache canonico (cv_all.csv + sarima_var.csv) via load_config e com o
calculo compartilhado em forecasting.precisao (mesma fonte usada pelo
scripts/check_winrates.py). Roda no venv principal (statsmodels + pmdarima
para a via anual-direta; matplotlib).

Uso:  python scripts/fig_hierarquia_temporal.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator

from forecasting.config import MUNITAX_BLUE, load_config
from forecasting.plotting import (
    BAR_COMPARISON,
    BAR_LABEL_SIZE,
    pixel_scale,
    rounded_bar,
    save_figure,
    setup_matplotlib_thesis,
    style_axis,
)
from forecasting.precisao import hierarquia_anual, hierarquia_mensal

SER_LABEL = {"salvador-IPTU": "Sal.\nIPTU", "salvador-ISSQN": "Sal.\nISSQN",
             "camacari-IPTU": "Cam.\nIPTU", "camacari-ISSQN": "Cam.\nISSQN",
             "ilheus-IPTU": "Ilh.\nIPTU", "ilheus-ISSQN": "Ilh.\nISSQN"}
SERIES = list(SER_LABEL)
FAMS = ["ARIMA", "ETS", "Naive/RW"]
FAM_LABEL = {"ARIMA": "ARIMA", "ETS": "ETS", "Naive/RW": "Naïve/RW"}


def _grupo_barras(ax, centros, vm, va, w, ymax):
    """Par de barras (mensal-agregada azul, anual-direta cinza claro) por
    grupo, arredondadas e sem moldura; rotulo de valor acima de cada uma."""
    ax.set_ylim(0, ymax)
    scale = pixel_scale(ax)
    for c, m, a in zip(centros, vm, va):
        if not np.isnan(m):
            rounded_bar(ax, c - w, 0, w, m, MUNITAX_BLUE, scale=scale)
            ax.text(c - w / 2, m + ymax * 0.02, f"{m:.1f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=BAR_LABEL_SIZE,
                    fontweight="semibold", color="#202124")
        if not np.isnan(a):
            rounded_bar(ax, c, 0, w, a, BAR_COMPARISON, scale=scale)
            ax.text(c + w / 2, a + ymax * 0.02, f"{a:.1f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=BAR_LABEL_SIZE,
                    fontweight="semibold", color="#5F6368")


def main() -> int:
    cfg = load_config()
    setup_matplotlib_thesis()
    mens = hierarquia_mensal(cfg)
    ann = hierarquia_anual(cfg)

    m_by = mens.groupby("fam")["err"].mean()
    a_by = ann.groupby("fam")["err"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.6), sharey=False)
    w = 0.34

    ax = axes[0]
    x = np.arange(len(FAMS))
    vm = [m_by.get(f, np.nan) for f in FAMS]
    va = [a_by.get(f, np.nan) for f in FAMS]
    ax.set_xlim(-0.6, len(FAMS) - 0.4)
    _grupo_barras(ax, x, vm, va, w, max(max(vm), max(va)) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in FAMS])
    ax.set_ylabel("Erro anual médio (%)")
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_title("Por família de modelo")
    style_axis(ax)
    ax.tick_params(axis="x", length=0.0)
    leg = [FancyBboxPatch((0, 0), 1, 1,
                          boxstyle="round,pad=0,rounding_size=0.12",
                          fc=MUNITAX_BLUE, ec="none", label="Mensal-agregada"),
           FancyBboxPatch((0, 0), 1, 1,
                          boxstyle="round,pad=0,rounding_size=0.12",
                          fc=BAR_COMPARISON, ec="none", label="Anual-direta")]
    # Legenda horizontal ancorada ao eixo (nao a figura): centrada sobre a
    # area de plotagem, alinhada com titulos e barras.
    axes[0].legend(handles=leg, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.12), frameon=False,
                   handlelength=1.1, handleheight=1.1, columnspacing=1.5,
                   handletextpad=0.5)

    # painel B: melhor de cada via por serie
    ax = axes[1]
    mb = (mens.groupby(["serie", "modelo"])["err"].mean()
          .groupby("serie").min().reindex(SERIES))
    ab = (ann.groupby(["serie", "fam"])["err"].mean()
          .groupby("serie").min().reindex(SERIES))
    xs = np.arange(len(SERIES))
    ax.set_xlim(-0.6, len(SERIES) - 0.4)
    _grupo_barras(ax, xs, mb.values, ab.values, w, max(mb.max(), ab.max()) * 1.22)
    ax.set_xticks(xs)
    ax.set_xticklabels([SER_LABEL[s] for s in SERIES])
    ax.set_ylabel("Erro anual médio (%)")
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_title("Melhor de cada via, por série")
    style_axis(ax)
    ax.tick_params(axis="x", length=0.0)

    out = save_figure(fig, "fig_hierarquia_temporal", cfg.figures_dir_abs)
    plt.close(fig)
    print(f"  + {out}  (ARIMA: mensal {m_by['ARIMA']:.1f} vs anual "
          f"{a_by['ARIMA']:.1f}; painel B medio: mensal {mb.mean():.1f} vs "
          f"anual {ab.mean():.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
