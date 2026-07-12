"""Carregamento da configuracao do pipeline a partir de .tcc-pipeline.json.

Permite que notebooks e scripts saibam, por convencao unificada:
- Onde esta o repo do TCC (para gravar tabelas/figuras).
- Onde estao os dados crus do siconfi-collector (CSVs coletados da API).
- Onde gravar previsoes geradas (analysis/data/forecasts/).
- Quais municipios e tributos compoem a amostra.
- Janela COVID, base do IPCA, horizontes de previsao.

A funcao `load_config()` busca o arquivo subindo ate encontrar
`.tcc-pipeline.json` ou em paths conhecidos. O config canonico e o
`.tcc-pipeline.json` da RAIZ do repositorio, com paths RELATIVOS a essa
raiz (portavel para qualquer clone); paths relativos sao resolvidos a
partir da pasta onde o arquivo de config esta. Um config alternativo com
paths absolutos pode ser passado explicitamente a `load_config(path)`.
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
    config_file = (Path(path) if path else _find_config_file()).resolve()
    # utf-8-sig: tolera BOM (editores/shells Windows costumam inseri-lo).
    raw: dict[str, Any] = json.loads(config_file.read_text(encoding="utf-8-sig"))

    municipalities = {
        key: Municipality(key=key, **data)
        for key, data in raw["municipalities"].items()
    }

    # Paths relativos no JSON sao resolvidos a partir da pasta do proprio
    # arquivo de config. Assim o .tcc-pipeline.json versionado na raiz do repo
    # funciona em qualquer clone; um config externo com paths absolutos
    # (ex.: maquina de desenvolvimento) continua funcionando como antes.
    base = config_file.parent

    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p).resolve()

    return PipelineConfig(
        tcc_root=_resolve(raw["tcc_root"]),
        tcc_figures_dir=Path(raw["tcc_figures_dir"]),
        tcc_tables_dir=Path(raw["tcc_tables_dir"]),
        analysis_root=_resolve(raw["analysis_root"]),
        siconfi_root=_resolve(raw["siconfi_root"]),
        siconfi_data_dir=_resolve(raw["siconfi_data_dir"]),
        forecasts_dir=_resolve(raw["forecasts_dir"]),
        municipalities=municipalities,
        tributos=list(raw["tributos"]),
        horizons=list(raw["horizons"]),
        sample_window=SampleWindow(**raw["sample_window"]),
        covid_period=CovidPeriod(**raw["covid_period"]),
        anomalies=[Anomaly(**a) for a in raw.get("anomalies", [])],
        ipca_base_month=raw["ipca_base_month"],
    )


# =============================================================================
# FONTE UNICA DE VERDADE: identidade dos modelos, cor, rotulos e formatacao.
# Antes duplicados em plotting/model_reports/evaluation/benchmarks/generalization
# /tex_export/eda. Tudo abaixo e a unica definicao; os demais modulos importam.
# =============================================================================

# Cor primaria da marca Munitax (plataforma de destino aplicado; ver Cap. 1 e 7).
# Esta cor e a ancora visual do documento: figuras, tabelas, diagramas e
# destaques derivam dela para manter uma identidade unica.
MUNITAX_BLUE = "#0582FF"

# Paleta categorica editorial, ancorada no azul Munitax. A ordem dos indices e
# preservada para retrocompatibilidade com scripts de figuras; cada modelo tem
# uma cor fixa em MODEL_COLORS.
THESIS_PALETTE = [
    "#9AA0A6",  # cinza neutro (baseline / anual-direto)
    "#34A853",  # verde frio (ETS)
    "#0582FF",  # azul Munitax (SARIMA / via mensal-agregada)
    "#7C3AED",  # indigo (Prophet)
    "#F29900",  # ambar controlado (Ensemble)
    "#12B5CB",  # ciano (Theta) -- distinto do azul Munitax do SARIMA
    "#202124",  # tinta principal
    "#E8F0FE",  # azul claro estrutural
]

# Ordem canonica dos modelos: baseline Naive + tres formais originais + Theta
# (parcimonioso, M3) + Ensemble (combinacao, ultimo por ser agregador).
MODEL_ORDER = ["Naive", "ETS", "SARIMA", "Prophet", "Theta", "Ensemble"]

# Rotulo LaTeX para as TABELAS .tex (Naive -> Na\"ive, acento via comando \").
MODEL_TEX = {"Naive": "Na\\\"ive", "ETS": "ETS", "SARIMA": "SARIMA",
             "Prophet": "Prophet", "Theta": "Theta", "Ensemble": "Ensemble"}

# Rotulo de EXIBICAO para matplotlib/figuras (unicode).
MODEL_LABELS = {"Naive": "Naïve", "ETS": "ETS", "SARIMA": "SARIMA",
                "Prophet": "Prophet", "Theta": "Theta", "Ensemble": "Ensemble"}

# Cor canonica por modelo, derivada da paleta Munitax (SEM hex solto): cada modelo
# tem UMA cor, usada em TODAS as figuras para o leitor associar cor a modelo.
#   Naive=neutro frio; SARIMA=azul Munitax; ETS=verde frio; Prophet=indigo;
#   Theta=ciano; Ensemble=âmbar. O quase-preto fica reservado para a serie
#   realizada.
MODEL_COLORS = {
    "Naive":    THESIS_PALETTE[0],
    "ETS":      THESIS_PALETTE[1],
    "SARIMA":   THESIS_PALETTE[2],
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


# ---------- Estilo de tabela do documento (identidade visual munitax) --------

def styled_table(*, gerado_por: str, caption: str, label: str, colspec: str,
                 header: list[str], rows: list[str], fonte: str,
                 footnote: str | None = None, stripe: bool = True,
                 size: str = "small", arraystretch: str = "1.32",
                 placement: str = "htbp", floating: bool = True,
                 width: str = "\\textwidth") -> str:
    r"""Monta uma tabela na CASA DE ESTILO do documento (FONTE UNICA do estilo).

    Largura total (``tabularx`` + ``\linewidth``), cabecalho claro, zebra
    discreta opcional, borda cinza fina e entrelinha compacta. As cores/macros
    (``tblHeader``, ``tblStripe``, ``tblBorder``, ``\thd``) vem do preambulo do
    ``main.tex``.

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
    # Apos \end{tcolorbox} estamos em modo vertical; \fonte (e \\) exigem uma
    # "linha". Com nota de rodape, o texto da nota fornece essa linha; sem nota,
    # \leavevmode a cria. Por isso NAO se usa \\ aqui.
    foot = (f"\\vspace{{3pt}}\n{{\\RaggedRight\\footnotesize\\textit{{Nota:}} "
            f"{footnote}\\par}}\n") if footnote else ""
    src = (f"\\fonte{{{fonte}}}\n" if footnote
           else f"\\leavevmode\\fonte{{{fonte}}}\n")
    body = "\n".join(rows)
    # Cartao de cantos arredondados: SEM padding interno (left/right/top/bottom=0)
    # para o cabecalho sangrar ate a moldura (a folga lateral do texto vem do
    # \tabcolsep das colunas de borda, ja que removemos o @{}); "clip upper"
    # recorta o conteudo a moldura -> cantos do cabecalho arredondados, sem
    # "dentes" quadrados. NAO ha espaco branco acima do cabecalho.
    box = ("\\begin{tcolorbox}[enhanced,colback=white,colframe=tblBorder,"
           "boxrule=0.45pt,arc=3pt,boxsep=0pt,left=0pt,right=0pt,top=0pt,"
           f"bottom=0pt,width={width},clip upper,before skip=10pt,after skip=3pt]")
    if floating:
        begin = (
            f"\\begin{{table}}[{placement}]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        )
        end = "\\end{table}\n"
    else:
        begin = (
            "\\par\\begingroup\\centering\n"
            "\\phantomsection\n"
            "\\refstepcounter{table}\n"
            f"\\label{{{label}}}\n"
            f"\\addcontentsline{{lot}}{{table}}{{\\protect\\numberline{{\\thetable}}{{{caption}}}}}\n"
            f"{{\\normalfont\\normalsize Tabela~\\thetable{{}} -- {caption}\\par}}\n"
        )
        end = "\\par\\endgroup\n"
    return (
        f"% Gerado por {gerado_por} -- nao editar a mao (estilo: config.styled_table).\n"
        f"{begin}"
        f"{box}\n"
        f"\\{size}\n"
        f"\\setlength{{\\tabcolsep}}{{6pt}}\\renewcommand{{\\arraystretch}}{{{arraystretch}}}\n"
        f"{zebra}"
        f"\\begin{{tabularx}}{{\\linewidth}}{{{colspec}}}\n"
        f"{head}\n"
        f"{body}\n"
        "\\end{tabularx}\n"
        "\\end{tcolorbox}\n"
        f"{foot}"
        f"{src}"
        f"{end}"
    )


# ---------- Chaves de serie derivadas do cfg (sem listas cravadas) -----------

def series_keys(cfg: PipelineConfig) -> list[tuple[str, str, str]]:
    """As seis (ou N) series como ``(municipio_key, municipio_nome, tributo)`` na
    ordem canonica (municipios do cfg x tributos do cfg). Substitui as listas das
    seis series antes recriadas a mao em varias funcoes."""
    return [(m.key, m.name, t)
            for m in cfg.municipalities.values()
            for t in cfg.tributos]


def mun_label(cfg: PipelineConfig) -> dict[str, str]:
    """Mapa ``municipio_key -> nome de exibicao``, derivado do cfg."""
    return {m.key: m.name for m in cfg.municipalities.values()}
