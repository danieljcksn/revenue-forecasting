"""Estado atual do problema: distribuicao do erro anual da projecao oficial
(LOA) das prefeituras baianas, por faixa, contra o Ensemble. Dados da coleta
estadual (1.556 anos-serie confrontaveis)."""
import sys
from pathlib import Path
ROOT = Path(r"C:/Users/Dan/Desktop/final-paper/final-paperz/analysis")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bahia"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import tcc_figures as T
from forecasting.config import load_config, MUNITAX_BLUE
from forecasting.plotting import (setup_matplotlib_thesis, save_figure, style_axis,
                                  pixel_scale, rounded_bar)

cfg = load_config()
setup_matplotlib_thesis()

m0 = T._confrontos()
m = m0[m0["modelo"] == "Ensemble"].dropna(subset=["erro_pct_prefeitura", "err"]).copy()
pref = m["erro_pct_prefeitura"].to_numpy()
ens = m["err"].to_numpy()
n = len(m)

bands = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9)]
labels = ["até 10%", "10 a 20%", "20 a 30%", "30 a 50%", "mais de 50%"]
def shares(x):
    return [((x >= lo) & (x < hi)).mean() * 100 for lo, hi in bands]
p_sh, e_sh = shares(pref), shares(ens)

PREF_C = "#9AA0A6"
ENS_C = MUNITAX_BLUE
fig, ax = plt.subplots(figsize=(6.0, 3.0))
scale = pixel_scale(ax)
x = np.arange(len(bands))
w = 0.38
ymax = max(max(p_sh), max(e_sh)) * 1.16
for xi, (p, e) in enumerate(zip(p_sh, e_sh)):
    rounded_bar(ax, xi - w, 0, w, p, PREF_C, scale=scale)
    rounded_bar(ax, xi, 0, w, e, ENS_C, scale=scale)
    ax.text(xi - w / 2, p + ymax * 0.02, f"{p:.0f}%", ha="center", va="bottom",
            fontsize=7.3, color="#5F6368", fontweight="semibold")
    ax.text(xi + w / 2, e + ymax * 0.02, f"{e:.0f}%", ha="center", va="bottom",
            fontsize=7.3, color=ENS_C, fontweight="semibold")

ax.set_xlim(-0.7, len(bands) - 0.3)
ax.set_ylim(0, ymax)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Erro anual absoluto da projeção (%), 2021 a 2025")
ax.set_ylabel("Anos-série (%)")
ax.grid(True, axis="y", color="#E8EAED", lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
style_axis(ax)

leg = [
    FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.12",
                   fc=PREF_C, ec="none", label="Projeção oficial (LOA)"),
    FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.12",
                   fc=ENS_C, ec="none", label="Ensemble"),
]
fig.legend(handles=leg, loc="outside upper center", ncol=2, fontsize=7.6,
           frameon=False, handlelength=1.1, handleheight=1.1, columnspacing=1.6,
           handletextpad=0.5)
out = save_figure(fig, "fig_erro_prefeituras", cfg.figures_dir_abs)
plt.close(fig)
hi_pref = (pref > 30).mean() * 100
print("salvo:", out, "| n =", n, "| pref>30%:", round(hi_pref), "| pref shares:",
      [round(s) for s in p_sh], "| ens shares:", [round(s) for s in e_sh])
