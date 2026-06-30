"""Estilo visual das figuras do TCC --- identidade editorial e coerente.

Define a "casa de estilo" das figuras: tipografia sem serifa limpa, muito
espaco em branco, pouca tinta-nao-dado (sem molduras superiores/laterais, grade
horizontal tenue), tracos confiantes e a cor-marca munitax (#0582FF) como
identidade (o baseline Naive e os realces). Toda figura do Cap. 5 chama
`setup_matplotlib_thesis()` antes de plotar e `save_figure()` ao gravar.

API publica (estavel; nao remover): setup_matplotlib_thesis, save_figure,
model_boxplot, panel_3x2, e as constantes REALIZED_INK/BASELINE_GREY. As cores e
rotulos por modelo vem do config (reexportados aqui para o codigo legado).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# FONTE UNICA de cor/rotulo: cores/rotulos por modelo + azul-marca vem do config.
from forecasting.config import (  # noqa: F401  (reexport p/ codigo legado)
    MODEL_COLORS,
    MODEL_LABELS,
    MUNITAX_BLUE,
    THESIS_PALETTE,
)

# Largura util do texto (A4, margens ABNT ~ 6 in). Figuras escalam sobre isso.
THESIS_TEXT_WIDTH_IN = 6.0

# --- Paleta bege da casa de estilo (DERIVA do tema LaTeX: themeAccent/Border/
#     Surface/Axis em main.tex). Mesma tinta e mesmos eixos das figuras TikZ do
#     Cap. 3/4, para que as figuras matplotlib do Cap. 5 leiam como o mesmo
#     documento. Cores de modelo (THESIS_PALETTE) permanecem os acentos de dados. ---
_INK = "#45413A"        # themeAccent: titulos e rotulos de eixo (tinta quente)
_INK_SOFT = "#6E695E"   # numeros dos ticks (cinza-quente)
_SPINE = "#B4B0A6"      # themeAxis: moldura/eixo base
_GRID = "#E7E3DA"       # grade horizontal (bege bem tenue)

# Serie realizada e linha de referencia: neutras, nunca uma cor de modelo.
REALIZED_INK = "#33302B"    # serie realizada (quase-preto quente, protagonista)
REALIZED_GREY = REALIZED_INK  # alias retrocompat
BASELINE_GREY = "#B4B0A6"   # themeAxis: linha de referencia (MASE=1), tracejada

# Pilha de fontes sem serifa (Arial/Helvetica no Windows/macOS; DejaVu no Linux).
_SANS = ["Arial", "Helvetica Neue", "Helvetica", "Segoe UI", "DejaVu Sans"]


def setup_matplotlib_thesis() -> None:
    """Aplica a casa de estilo do TCC ao matplotlib (idempotente).

    Estilo editorial: sem serifa, sem molduras superiores/direita/esquerda,
    grade horizontal tenue atras dos dados, tipografia confiante e fundo
    transparente (integra-se ao papel). Fixa tambem o backend Agg (FONTE UNICA).
    """
    import matplotlib as mpl
    from cycler import cycler

    mpl.use("Agg")  # backend nao-interativo: FONTE UNICA do use("Agg")
    mpl.rcParams.update({
        # --- tipografia ---
        "font.family": "sans-serif",
        "font.sans-serif": _SANS,
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelweight": "regular",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
        # --- legenda enxuta (sem moldura, horizontal) ---
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handleheight": 0.9,
        "legend.columnspacing": 1.5,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.0,
        "legend.labelcolor": _INK,
        # --- figura / layout ---
        "figure.figsize": (THESIS_TEXT_WIDTH_IN, THESIS_TEXT_WIDTH_IN * 0.60),
        "figure.dpi": 110,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.transparent": True,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.08,
        "figure.constrained_layout.w_pad": 0.08,
        "figure.constrained_layout.hspace": 0.06,
        "figure.constrained_layout.wspace": 0.06,
        # --- cores e ciclo ---
        "axes.prop_cycle": cycler(color=THESIS_PALETTE),
        "axes.edgecolor": _SPINE,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "text.color": _INK,
        "axes.axisbelow": True,
        # --- molduras: so a base, fina; sem topo/direita/esquerda ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        # --- grade horizontal tenue ---
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": _GRID,
        "grid.alpha": 1.0,
        "grid.linewidth": 0.8,
        # --- ticks discretos (sem marca no y; o grid ja orienta) ---
        "xtick.color": _SPINE,
        "ytick.color": _SPINE,
        "xtick.labelcolor": _INK_SOFT,
        "ytick.labelcolor": _INK_SOFT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 0.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.0,
        "xtick.major.pad": 4,
        "ytick.major.pad": 3,
        # --- tracos ---
        "lines.linewidth": 1.7,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.antialiased": True,
        "patch.linewidth": 0.0,
        "patch.antialiased": True,
    })


def style_axis(ax: Any) -> None:
    """Polimento por eixo: afasta a base, suaviza a grade e arruma os ticks.
    Chamar depois de plotar (idempotente)."""
    ax.spines["bottom"].set_color(_SPINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors=_SPINE, labelcolor=_INK_SOFT, length=3.0)
    ax.tick_params(axis="y", length=0.0, labelcolor=_INK_SOFT)
    ax.set_axisbelow(True)


def thousands_formatter(decimals: int = 0):
    """Formatador de eixo em convencao BR (milhar com ponto, decimal com virgula).
    Use em valores ja em R$ milhoes para eixos legiveis."""
    from matplotlib.ticker import FuncFormatter

    def _fmt(x, _pos):
        s = f"{x:,.{decimals}f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    return FuncFormatter(_fmt)


def clean_legend(fig_or_ax: Any, handles=None, labels=None, *, ncol: int = 5,
                 loc: str = "outside upper center", **kwargs):
    """Legenda da casa de estilo: horizontal, sem moldura, acima do conjunto."""
    common = dict(frameon=False, ncol=ncol, loc=loc,
                  handlelength=1.5, columnspacing=1.5, handletextpad=0.5,
                  fontsize=8.5, labelcolor=_INK)
    common.update(kwargs)
    if handles is not None:
        return fig_or_ax.legend(handles, labels, **common)
    return fig_or_ax.legend(**common)


def model_boxplot(ax: Any, data: list, models: list[str], *, ref: float = 1.0):
    """Boxplot da casa de estilo: uma caixa por modelo na cor canonica, mediana
    branca, bigodes/\\emph{caps} discretos e a linha de referencia (MASE=1)
    neutra ao fundo. Mantem estilo identico no nucleo e na generalizacao."""
    if ref is not None:
        ax.axhline(ref, color=BASELINE_GREY, lw=1.0, ls=(0, (5, 4)), zorder=0)
    bp = ax.boxplot(
        data, tick_labels=[MODEL_LABELS[m] for m in models],
        showfliers=False, patch_artist=True, widths=0.60,
        medianprops=dict(color="white", lw=1.6, solid_capstyle="butt"),
        whiskerprops=dict(color="#A39E92", lw=0.9),
        capprops=dict(color="#A39E92", lw=0.9),
        boxprops=dict(linewidth=0),
        zorder=3,
    )
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.92)
    style_axis(ax)
    ax.tick_params(axis="x", length=0.0)  # rotulos de modelo nao precisam de tick
    return bp


def save_figure(fig: Any, name: str, output_dir: Path | str) -> Path:
    """Grava a figura em PDF vetorial no diretorio do TCC (fundo transparente).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : nome canonico (e.g. "fig_serie_temporal"); sufixo .pdf adicionado.
    output_dir : geralmente <tcc_root>/figures/generated/.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".pdf"):
        name = f"{name}.pdf"
    path = output_dir / name
    fig.savefig(path)
    return path


def panel_3x2(figsize_scale: float = 1.0, *, ratio: float = 0.92) -> Any:
    """Figura 3x2 (municipio x tributo) padronizada da casa de estilo.

    Linhas = municipios [Salvador, Camaçari, Ilheus]; colunas = tributos
    [IPTU, ISSQN]. Eixo x compartilhado (datas); cada coluna com seu y.
    Retorna (fig, axes_2d).
    """
    import matplotlib.pyplot as plt

    width = THESIS_TEXT_WIDTH_IN * figsize_scale
    height = width * ratio
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(width, height), sharex=True, sharey=False,
    )
    municipios = ["Salvador", "Camaçari", "Ilhéus"]
    tributos = ["IPTU", "ISSQN"]
    for i, m in enumerate(municipios):
        axes[i, 0].set_ylabel(m, fontweight="bold")
    for j, t in enumerate(tributos):
        axes[0, j].set_title(t)
    for ax in axes.flat:
        style_axis(ax)
    return fig, axes
