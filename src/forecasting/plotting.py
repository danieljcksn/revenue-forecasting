"""Estilo visual das figuras do TCC --- identidade editorial e coerente.

Define a "casa de estilo" das figuras: tipografia sem serifa limpa, muito
espaco em branco, pouca tinta-nao-dado (sem molduras superiores/laterais, grade
horizontal tenue), tracos confiantes e a cor-marca Munitax (#0582FF) como
identidade editorial. Toda figura do Cap. 5 chama
`setup_matplotlib_thesis()` antes de plotar e `save_figure()` ao gravar.

API publica: setup_matplotlib_thesis, save_figure, style_axis, br_axis,
clean_legend e as constantes REALIZED_INK/BASELINE_GREY. As cores e rotulos
por modelo vem do config (reexportados aqui por conveniencia).
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

# --- Paleta Munitax da casa de estilo (DERIVA do tema LaTeX: themeAccent/
#     Border/Surface/Axis em main.tex). Mesma tinta e mesmos eixos das figuras
#     TikZ do Cap. 3/4, para que as figuras matplotlib leiam como um unico
#     documento. Cores de modelo (THESIS_PALETTE) permanecem os acentos de dados. ---
_INK = "#202124"        # themeAccent: titulos e rotulos de eixo
_INK_SOFT = "#5F6368"   # numeros dos ticks e texto secundario
_SPINE = "#DADCE0"      # themeAxis: eixo base/divisorias
_GRID = "#E8EAED"       # grade horizontal cinza bem tenue

# Serie realizada e linha de referencia: neutras, nunca uma cor de modelo.
REALIZED_INK = "#202124"    # serie realizada (quase-preto frio, protagonista)
BASELINE_GREY = "#B7BCC2"   # linha de referencia (MASE=1): visivel, ainda neutra

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
        "font.size": 9.0,
        "axes.titlesize": 9.2,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.4,
        "axes.labelweight": "regular",
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.7,
        "figure.titlesize": 9.4,
        "figure.titleweight": "semibold",
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
        "savefig.pad_inches": 0.06,
        "savefig.transparent": True,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.12,
        "figure.constrained_layout.w_pad": 0.12,
        "figure.constrained_layout.hspace": 0.10,
        "figure.constrained_layout.wspace": 0.10,
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
        "grid.linewidth": 0.6,
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
        "lines.linewidth": 1.55,
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


def br_decimal_formatter(decimals: int = 1):
    """Formatador de ticks com virgula decimal BR (ex.: 0,8 / 1,0).

    TODO eixo numerico nao-inteiro do documento usa este formatador: a
    convencao decimal das figuras e a MESMA do texto e das tabelas."""
    from matplotlib.ticker import FuncFormatter

    def _fmt(x, _pos):
        return f"{x:.{decimals}f}".replace(".", ",")
    return FuncFormatter(_fmt)


def br_axis(ax: Any, axis: str = "y", decimals: int = 1,
            step: float | None = None) -> None:
    """Aplica ao eixo o locator regular e a virgula decimal BR.

    ``step`` fixa o espacamento dos ticks (MultipleLocator); sem ele, mantem o
    locator vigente e apenas troca o formato. Evita o artefato de passos
    irregulares (0,0 / 0,2 / 0,5 / 0,8) que o locator automatico arredondado a
    1 casa produz."""
    from matplotlib.ticker import MultipleLocator

    target = ax.yaxis if axis == "y" else ax.xaxis
    if step is not None:
        target.set_major_locator(MultipleLocator(step))
    target.set_major_formatter(br_decimal_formatter(decimals))


def clean_legend(fig_or_ax: Any, handles=None, labels=None, *, ncol: int = 5,
                 loc: str = "outside upper center", **kwargs):
    """Legenda da casa de estilo: horizontal, sem moldura, acima do conjunto."""
    common = dict(frameon=False, ncol=ncol, loc=loc,
                  handlelength=1.5, columnspacing=1.5, handletextpad=0.5,
                  fontsize=7.7, labelcolor=_INK)
    common.update(kwargs)
    if handles is not None:
        return fig_or_ax.legend(handles, labels, **common)
    return fig_or_ax.legend(**common)


# --- Sistema de barras (FONTE UNICA) -----------------------------------------
# Cinza de comparacao (barra secundaria), no espirito dos charts do Google:
# solido, claro, SEM moldura. Distinto do #9AA0A6 (Naive) por ser mais claro.
BAR_COMPARISON = "#DDE1E6"   # barra de comparacao/contrafactual
BAR_LABEL_SIZE = 7.4         # tamanho unico dos rotulos de valor sobre barras


def pixel_scale(ax: Any) -> tuple[float, float]:
    """(px, py) = pixels por unidade de dado em x e y, apos o layout. Calcular
    UMA vez por eixo (depois de fixar os limites) e reutilizar em rounded_bar."""
    ax.figure.canvas.draw()
    p0 = ax.transData.transform((0, 0))
    return (ax.transData.transform((1, 0))[0] - p0[0],
            ax.transData.transform((0, 1))[1] - p0[1])


def rounded_bar(ax: Any, x: float, y: float, width: float, height: float,
                facecolor: str, *, scale: tuple[float, float], radius_px: float = 3.0,
                hatch: str | None = None, edgecolor: str = "none",
                zorder: float = 3, alpha: float = 1.0) -> None:
    """Barra com cantos arredondados de raio CONSTANTE em pixels (independe da
    escala dos eixos), no estilo dos charts do Google. Serve barras verticais
    (chame com y=0, height=valor) e horizontais (x=0, width=valor). ``scale``
    vem de pixel_scale(ax); ``radius_px`` e o raio comum a TODAS as barras do
    documento."""
    import matplotlib.patches as mpatches

    px, py = scale
    r_x = radius_px / abs(px)
    lw = 0.9 if edgecolor != "none" else 0.0
    patch = mpatches.FancyBboxPatch(
        (x, y), max(width, r_x * 2), max(height, r_x * 2 * abs(px / py)),
        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=r_x),
        mutation_aspect=px / py, facecolor=facecolor, edgecolor=edgecolor,
        hatch=hatch, linewidth=lw, joinstyle="round", alpha=alpha,
        zorder=zorder, clip_on=False)
    ax.add_patch(patch)


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

