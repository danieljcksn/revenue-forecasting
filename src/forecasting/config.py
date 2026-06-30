"""Pipeline configuration loader.

The `.tcc-pipeline.json` file records manuscript output paths, collector data
paths, forecast-cache locations, sample definitions, and shared parameters.
`load_config()` searches upward from the current directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Municipality:
    key: str
    cod_ibge: int
    name: str
    uf: str


@dataclass(frozen=True)
class CovidPeriod:
    start: str
    end: str


@dataclass(frozen=True)
class SampleWindow:
    start: str  # ex.: "2015-01"
    end: str    # ex.: "2025-12"


@dataclass(frozen=True)
class Anomaly:
    """Anomalia de dados conhecida que exige tratamento explicito."""
    municipio: str
    tributo: str
    year: int
    treatment: str
    note: str


@dataclass(frozen=True)
class PipelineConfig:
    tcc_root: Path
    tcc_figures_dir: Path
    tcc_tables_dir: Path
    analysis_root: Path
    siconfi_root: Path
    siconfi_data_dir: Path
    forecasts_dir: Path
    municipalities: dict[str, Municipality]
    tributos: list[str]
    horizons: list[int]
    sample_window: SampleWindow
    covid_period: CovidPeriod
    anomalies: list[Anomaly]
    ipca_base_month: str

    @property
    def figures_dir_abs(self) -> Path:
        return self.tcc_root / self.tcc_figures_dir

    @property
    def tables_dir_abs(self) -> Path:
        return self.tcc_root / self.tcc_tables_dir


def _find_config_file(start: Path | None = None) -> Path:
    here = Path(start) if start else Path.cwd()
    here = here.resolve()
    for candidate in (here, *here.parents):
        config = candidate / ".tcc-pipeline.json"
        if config.exists():
            return config
        nested = candidate / "tcc-latex" / ".tcc-pipeline.json"
        if nested.exists():
            return nested
    raise FileNotFoundError(
        ".tcc-pipeline.json nao encontrado a partir de "
        f"{here}. Verifique se voce esta dentro do wrapper do projeto."
    )


def load_config(path: Path | str | None = None) -> PipelineConfig:
    config_file = Path(path) if path else _find_config_file()
    raw: dict[str, Any] = json.loads(config_file.read_text(encoding="utf-8"))

    municipalities = {
        key: Municipality(key=key, **data)
        for key, data in raw["municipalities"].items()
    }

    return PipelineConfig(
        tcc_root=Path(raw["tcc_root"]),
        tcc_figures_dir=Path(raw["tcc_figures_dir"]),
        tcc_tables_dir=Path(raw["tcc_tables_dir"]),
        analysis_root=Path(raw["analysis_root"]),
        siconfi_root=Path(raw["siconfi_root"]),
        siconfi_data_dir=Path(raw["siconfi_data_dir"]),
        forecasts_dir=Path(raw["forecasts_dir"]),
        municipalities=municipalities,
        tributos=list(raw["tributos"]),
        horizons=list(raw["horizons"]),
        sample_window=SampleWindow(**raw["sample_window"]),
        covid_period=CovidPeriod(**raw["covid_period"]),
        anomalies=[Anomaly(**a) for a in raw.get("anomalies", [])],
        ipca_base_month=raw["ipca_base_month"],
    )


# =============================================================================
# Single source of truth: model identity, colors, labels, and formatting.
# =============================================================================

# Primary product color. Kept separate from model colors.
MUNITAX_BLUE = "#0582FF"

# Warm categorical palette for model series.
THESIS_PALETTE = [
    "#9C8466",  # taupe/caqui (Naive, baseline neutro)
    "#C0703A",  # terracota (SARIMA)
    "#3F7268",  # verde-musgo (ETS)
    "#8A6A8E",  # malva (Prophet)
    "#A6552F",  # rust (reserva)
    "#6E5A7E",  # ameixa (reserva)
    "#45413A",  # tinta quente (reserva)
    "#B7A98C",  # areia (reserva)
]

# Canonical model order: baseline, individual models, aggregate model.
MODEL_ORDER = ["Naive", "ETS", "SARIMA", "Prophet", "Theta", "Ensemble"]

# LaTeX labels for generated tables.
MODEL_TEX = {"Naive": "Na\\\"ive", "ETS": "ETS", "SARIMA": "SARIMA",
             "Prophet": "Prophet", "Theta": "Theta", "Ensemble": "Ensemble"}

# Display labels for matplotlib figures.
MODEL_LABELS = {"Naive": "Naïve", "ETS": "ETS", "SARIMA": "SARIMA",
                "Prophet": "Prophet", "Theta": "Theta", "Ensemble": "Ensemble"}

# Canonical model colors used consistently across figures.
MODEL_COLORS = {
    "Naive":    THESIS_PALETTE[0],
    "ETS":      THESIS_PALETTE[2],
    "SARIMA":   THESIS_PALETTE[1],
    "Prophet":  THESIS_PALETTE[3],
    "Theta":    THESIS_PALETTE[5],
    "Ensemble": THESIS_PALETTE[4],
}


# ---------- Formatadores numericos brasileiros (FONTE UNICA) -----------------

def format_brl(value: float | int | None, decimals: int = 2) -> str:
    """Numero na convencao BR: milhar com ponto, decimal com virgula
    (ex.: ``1.234.567,89``). None/NaN -> ``--``."""
    if value is None or pd.isna(value):
        return "--"
    s = f"{float(value):,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def format_dec(value: float | int | None, decimals: int = 2) -> str:
    """Decimal BR SEM separador de milhar e SEM sufixo (ex.: ``0,73``).
    None/NaN -> ``--``. Substitui os ``f"{x:.Nf}".replace(".", ",")`` espalhados."""
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}".replace(".", ",")


def format_pct(value: float | None, decimals: int = 2) -> str:
    """Decimal BR com sufixo de porcentagem LaTeX (ex.: ``12,34\\%``)."""
    if value is None or pd.isna(value):
        return "--"
    return f"{format_dec(value, decimals)}\\%"


def format_int(value: int | None) -> str:
    """Inteiro com separador de milhar BR (ex.: ``1.234``)."""
    if value is None or pd.isna(value):
        return "--"
    return f"{int(value):,}".replace(",", ".")


# ---------- Generated-table style -------------------------------------------

def styled_table(*, gerado_por: str, caption: str, label: str, colspec: str,
                 header: list[str], rows: list[str], fonte: str,
                 footnote: str | None = None, stripe: bool = True,
                 size: str = "small") -> str:
    r"""Build a styled LaTeX table.

    Uses ``tabularx`` at full text width, a styled header, optional zebra
    stripes, thin rules, and slightly expanded line spacing.

    Parameters
    ----------
    colspec : colunas ``tabularx`` separadas por espaco, ex.: ``"L l R R R"``.
        Use ``L``/``R``/``C`` (flexiveis, definidas no preambulo) para a tabela
        preencher a largura do texto; ``l``/``r``/``c`` ficam na largura natural.
    header : celulas do cabecalho como texto cru (o estilo aplica ``\thd``).
    rows : linhas do corpo ja montadas (``"a & b & c \\"``).
    stripe : zebra discreta a partir da 1a linha do corpo (desligar em tabelas
        com subcabecalhos ``\multicolumn``).
    """
    head = ("\\rowcolor{tblHeader}\n"
            + " & ".join(f"\\thd{{{c}}}" for c in header) + " \\\\")
    zebra = "\\rowcolors{2}{white}{tblStripe}\n" if stripe else ""
    # After tcolorbox LaTeX is in vertical mode; \fonte needs a paragraph.
    foot = f"\\vspace{{3pt}}\n{{\\footnotesize {footnote}}}\n" if footnote else ""
    src = (f"\\fonte{{{fonte}}}\n" if footnote
           else f"\\leavevmode\\fonte{{{fonte}}}\n")
    body = "\n".join(rows)
    # No inner padding: header color reaches the rounded border cleanly.
    box = ("\\begin{tcolorbox}[enhanced,colback=white,colframe=tblBorder,"
           "boxrule=0.6pt,arc=4pt,boxsep=0pt,left=0pt,right=0pt,top=0pt,"
           "bottom=0pt,width=\\textwidth,clip upper,before skip=2pt,after skip=2pt]")
    return (
        f"% Gerado por {gerado_por} -- nao editar a mao (estilo: config.styled_table).\n"
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{box}\n"
        f"\\{size}\n"
        "\\setlength{\\tabcolsep}{6pt}\\renewcommand{\\arraystretch}{1.45}\n"
        f"{zebra}"
        f"\\begin{{tabularx}}{{\\linewidth}}{{{colspec}}}\n"
        f"{head}\n"
        f"{body}\n"
        "\\end{tabularx}\n"
        "\\end{tcolorbox}\n"
        f"{foot}"
        f"{src}"
        "\\end{table}\n"
    )


# ---------- Series keys derived from config ----------------------------------

def series_keys(cfg: PipelineConfig) -> list[tuple[str, str, str]]:
    """Return series as ``(municipio_key, municipio_nome, tributo)`` tuples."""
    return [(m.key, m.name, t)
            for m in cfg.municipalities.values()
            for t in cfg.tributos]


def mun_label(cfg: PipelineConfig) -> dict[str, str]:
    """Mapa ``municipio_key -> nome de exibicao``, derivado do cfg."""
    return {m.key: m.name for m in cfg.municipalities.values()}
