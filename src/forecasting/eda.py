"""Analise exploratoria - gera os artefatos da Secao 5.1 do TCC.

Cada funcao publica produz um artefato canonico (tabela .tex ou figura .pdf) e
retorna o path final. As tres tabelas compartilham o helper interno
``_write_table`` (estrutura ABNT ``\\begin{table}...\\fonte``); as figuras usam
``plotting.setup_matplotlib_thesis``. Escopo: estacionariedade por ADF + KPSS
(sem Phillips-Perron); estabilizacao de variancia por log (sem Box-Cox); sem
ajuste de dias uteis (confundente conhecida, nao tratada).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import PipelineConfig, format_brl
from forecasting.io import load_monthly_series, table_path, tributo_column
from forecasting.plotting import save_figure, setup_matplotlib_thesis

# Colunas identificadoras do monthly_revenue.csv (nao sao valores a deflacionar).
_ID_COLS = {"cod_ibge", "entity_name", "uf", "year", "month", "month_name", "date"}


def _load_ipca_index(analysis_root: Path | str) -> pd.Series:
    """Carrega o indice acumulado do IPCA de ``analysis/data/ipca_sgs433.csv``.

    O CSV (gerado a partir da serie SGS 433 do BCB --- variacao mensal % ---)
    tem colunas: year, month, date, ipca_var_pct, ipca_index,
    deflator_to_2025_12. O ``ipca_index`` e o produto acumulado
    ``I_t = I_{t-1} * (1 + var_t/100)`` com I anterior ao primeiro mes = 1.

    Retorna uma Series ``ipca_index`` indexada por ``pandas.Period`` mensal.
    """
    path = Path(analysis_root) / "data" / "ipca_sgs433.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nao encontrado. Baixe a serie SGS 433 do BCB "
            "(https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json), "
            "construa o indice acumulado I_t = I_{t-1}*(1+var_t/100) e salve com "
            "colunas: year, month, date, ipca_var_pct, ipca_index, "
            "deflator_to_2025_12 (ver o script que gerou este arquivo)."
        )
    ipca = pd.read_csv(path)
    period = pd.PeriodIndex(
        pd.to_datetime(ipca["year"].astype(str) + "-" + ipca["month"].astype(str) + "-01"),
        freq="M",
    )
    return pd.Series(ipca["ipca_index"].to_numpy(), index=period, name="ipca_index")


def prepare_series(
    cfg: PipelineConfig,
    impute: bool = True,
) -> dict[tuple[str, str], pd.Series]:
    """Constroi as seis series mensais (municipio x tributo) em reais constantes,
    prontas para a modelagem -- a FONTE UNICA de series da EDA, da modelagem e
    dos benchmarks. Etapas, por (municipio, tributo):

      1. Carregar o ``monthly_revenue.csv`` (saida do siconfi-collector).
      2. Deflacionar pelo IPCA (SGS 433 do BCB) para o mes-base ``cfg.ipca_base_month``.
      3. Recortar na janela ``cfg.sample_window`` (jan/2015 a dez/2025).
      4. ``asfreq("MS")``: um ponto por mes; meses ausentes viram ``NaN``.
      5. Se ``impute=True``, tratar as anomalias de ``cfg.anomalies`` -- hoje so o
         ISSQN de Camacari em 2016 (patamar ~75% abaixo, reclassificacao PCASP),
         substituido pela media do mesmo mes nos anos vizinhos (Naive sazonal).

    Use ``impute=False`` quando a anomalia deve permanecer VISIVEL
    (``time_series_panel``, ``coverage_table``); ``impute=True`` (default) para a
    modelagem e a ``stationarity_table``. Retorna ``dict[(mun_key, tributo) -> Series]``.
    """
    raw = load_monthly_series(cfg)
    defl = deflate_by_ipca(raw, base_month=cfg.ipca_base_month)
    start = pd.Timestamp(cfg.sample_window.start + "-01")
    end = pd.Timestamp(cfg.sample_window.end + "-01")

    anomaly_map = {(a.municipio, a.tributo): a for a in cfg.anomalies}
    out: dict[tuple[str, str], pd.Series] = {}
    for key, mun in cfg.municipalities.items():
        sub = defl[defl["cod_ibge"] == mun.cod_ibge].sort_values("date")
        for tributo in cfg.tributos:
            col = tributo_column(tributo)
            s = pd.Series(
                pd.to_numeric(sub[col], errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(sub["date"]).to_period("M").to_timestamp(),
                name=f"{mun.name}-{tributo}",
            )
            s = s.loc[(s.index >= start) & (s.index <= end)].asfreq("MS")
            anomaly = anomaly_map.get((key, tributo))
            if impute and anomaly is not None:
                s = impute_anomalous_year(s, anomaly.year)
            out[(key, tributo)] = s
    return out


def deflate_by_ipca(
    df: pd.DataFrame,
    base_month: str,
    value_columns: list[str] | None = None,
    analysis_root: Path | str | None = None,
) -> pd.DataFrame:
    """Deflaciona valores nominais pelo IPCA (SGS 433 do BCB) para ``base_month``.

    Parameters
    ----------
    df : DataFrame com coluna ``date`` (datetime ou parseavel) e colunas de valor.
    base_month : ex.: ``"2025-12"`` --- todos os valores ficam em reais desse mes.
    value_columns : colunas a deflacionar; se None, todas as colunas numericas
        que nao forem identificadores conhecidos (cod_ibge, year, month, ...).
    analysis_root : raiz do projeto de analise (onde fica ``data/ipca_sgs433.csv``);
        se None, deduz a partir da localizacao deste modulo.

    Returns
    -------
    Copia de ``df`` com as colunas de valor deflacionadas (mesmos nomes):
    ``V_real,t = V_nominal,t * I_base / I_t``.
    """
    if analysis_root is None:
        analysis_root = Path(__file__).resolve().parents[2]  # src/forecasting/eda.py -> analysis/
    idx = _load_ipca_index(analysis_root)

    base_p = pd.Period(base_month, freq="M")
    if base_p not in idx.index:
        raise ValueError(
            f"base_month {base_month} fora do periodo coberto pelo IPCA "
            f"({idx.index.min()}..{idx.index.max()})"
        )

    out = df.copy()
    out_periods = pd.PeriodIndex(pd.to_datetime(out["date"]), freq="M")
    factor = float(idx.loc[base_p]) / idx.reindex(out_periods).to_numpy()

    if value_columns is None:
        value_columns = [
            c for c in out.columns
            if c not in _ID_COLS and pd.api.types.is_numeric_dtype(out[c])
        ]
    for c in value_columns:
        out[c] = pd.to_numeric(out[c], errors="coerce") * factor
    return out


def detect_outliers_stl_iqr(
    series: pd.Series,
    iqr_factor: float = 1.5,
) -> pd.Series:
    """Detecta outliers via STL + IQR no componente residual.

    Returns
    -------
    Boolean Series alinhada a `series`: True onde o ponto e outlier.
    """
    from statsmodels.tsa.seasonal import STL

    s = pd.Series(np.asarray(series, dtype=float), index=pd.DatetimeIndex(series.index))
    resid = STL(s, period=12, robust=True).fit().resid
    q1, q3 = np.percentile(resid.dropna(), [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    return (resid < lo) | (resid > hi)


def outlier_screen(cfg: PipelineConfig) -> Path:
    """Executa o rastreio de outliers (STL + IQR) sobre as seis series e grava
    um resumo reproduzivel em ``data/forecasts/outlier_screen.txt``.

    O rastreio e deliberadamente sensivel: sinaliza sobretudo os picos sazonais
    legitimos do IPTU no primeiro trimestre, que sao arrecadacoes reais e
    permanecem na serie. A inspecao confirma uma unica substituicao efetiva ---
    a anomalia declarada em ``cfg.anomalies`` (ISSQN de Camacari, 2016). Este
    passo torna o tratamento de outliers descrito na metodologia efetivamente
    executavel e auditavel, em vez de apenas declarado.
    """
    series = prepare_series(cfg, impute=False)
    declared = {(a.municipio, a.tributo): a.year for a in cfg.anomalies}
    lines = ["# Rastreio de outliers: STL (period=12, robust) + IQR (fator 1,5)",
             "# sobre o residuo. serie -> n_sinalizados : meses"]
    total = 0
    for (mk, trib), s in series.items():
        flags = detect_outliers_stl_iqr(s)
        n = int(flags.sum())
        total += n
        months = ", ".join(d.strftime("%Y-%m") for d in s.index[flags.to_numpy()])
        lines.append(f"{mk}-{trib} -> {n}: {months}")
    lines.append(f"# total sinalizado: {total}")
    lines.append(f"# substituicoes efetivas (anomalias confirmadas): {declared}")
    out = cfg.forecasts_dir / "outlier_screen.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def impute_anomalous_year(
    series: pd.Series,
    year: int,
    method: str = "naive_seasonal_from_adjacent",
) -> pd.Series:
    """Imputa um exercicio inteiro anomalo (nivel deslocado por 12 meses).

    Caso concreto: ISSQN de Camacari em 2016, cujo patamar mensal fica ~75%
    abaixo dos exercicios adjacentes (subnotificacao / reclassificacao PCASP).
    Diferente de `detect_outliers_stl_iqr`, que trata pontos isolados, esta
    funcao substitui os 12 meses do ano `year`.

    method:
      - "naive_seasonal_from_adjacent": cada mes m de `year` recebe a media
        do mesmo mes em year-1 e year+1 (quando disponiveis).
      - "drop": marca os 12 meses como NaN (a serie passa a ter um buraco
        anual, tratado a jusante como ausencia).

    Retorna uma nova Series; nao modifica `series` in-place. A localizacao e o
    metodo de imputacao devem ser registrados pelo chamador para a tabela de
    cobertura e para a analise de sensibilidade reportada nos resultados.
    """
    out = series.copy()
    idx = pd.DatetimeIndex(out.index)
    years = idx.year
    months = idx.month

    if method == "drop":
        out[years == year] = float("nan")
        return out

    if method != "naive_seasonal_from_adjacent":
        raise ValueError(f"metodo de imputacao desconhecido: {method!r}")

    for m in range(1, 13):
        neighbours = []
        for mask in (
            (years == year - 1) & (months == m),
            (years == year + 1) & (months == m),
        ):
            if mask.any():
                v = float(out[mask].iloc[0])
                if not np.isnan(v):  # vizinho ausente nao deve sobrescrever
                    neighbours.append(v)
        if not neighbours:
            continue  # sem vizinhos disponiveis; mantem o valor original do mes
        out.loc[(years == year) & (months == m)] = float(np.mean(neighbours))
    return out


def _write_table(
    cfg: PipelineConfig, *, name: str, gerado_por: str, caption: str, label: str,
    colspec: str, header, rows: list[str], fonte: str,
    footnote: str | None = None, tabularx: bool = True,
    stripe: bool = True, size: str = "small",
) -> Path:
    """Delegado fino para a casa de estilo unica (``config.styled_table``):
    tabela de largura total, cabecalho azul-marca munitax e zebra discreta.
    ``header`` aceita string (``"a & b \\\\"``) ou lista de celulas. ``tabularx``
    fica por compatibilidade, mas e ignorado (o estilo usa sempre tabularx)."""
    from forecasting.config import styled_table
    if isinstance(header, str):
        h = header.strip()
        if h.endswith("\\\\"):
            h = h[:-2]
        header = [c.strip() for c in h.split("&")]
    tex = styled_table(
        gerado_por=gerado_por, caption=caption, label=label, colspec=colspec,
        header=header, rows=rows, fonte=fonte, footnote=footnote,
        stripe=stripe, size=size)
    out = table_path(cfg, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def descriptive_stats_table(
    cfg: PipelineConfig,
    monthly_df: pd.DataFrame,
) -> Path:
    """Gera tab_descritivas.tex (Tabela 5.1.1).

    `monthly_df` deve estar JA DEFLACIONADO (passe o resultado de
    `deflate_by_ipca`). Itera (municipio, tributo) de `cfg`, calcula n, media,
    mediana, desvio-padrao, CV (%), minimo e maximo em R$ milhoes, e escreve a
    tabela LaTeX (formato ABNT, separador decimal virgula) em
    `cfg.tables_dir_abs / tab_descritivas.tex`. Retorna o path gerado.
    """
    muns = list(cfg.municipalities.values())
    rows: list[str] = []
    for i, mun in enumerate(muns):
        for tax in cfg.tributos:
            col = tributo_column(tax)
            s = pd.to_numeric(
                monthly_df.loc[monthly_df["cod_ibge"] == mun.cod_ibge, col],
                errors="coerce",
            ).dropna() / 1e6
            n = int(s.shape[0])
            mean = s.mean()
            cv = (s.std() / mean * 100.0) if mean else float("nan")
            rows.append(
                f"{mun.name} & {tax} & {n} & {format_brl(mean, 1)} & {format_brl(s.median(), 1)} "
                f"& {format_brl(s.std(), 1)} & {format_brl(cv, 1)} "
                f"& {format_brl(s.min(), 1)}--{format_brl(s.max(), 1)} \\\\"
            )
        if i < len(muns) - 1:
            rows.append(r"\addlinespace")

    base = cfg.ipca_base_month  # ex.: "2025-12"
    base_label = f"dez/{base[:4]}" if base.endswith("-12") else base
    return _write_table(
        cfg,
        name="tab_descritivas",
        gerado_por="eda.descriptive_stats_table",
        caption=(f"Estat\\'isticas descritivas das s\\'eries mensais "
                 f"(valores reais, base IPCA {base_label}, R\\$ milh\\~oes)"),
        label="tab:descritivas",
        colspec="l l C C C C C c",
        size="footnotesize",
        header=("Munic\\'ipio & Tributo & $n$ & M\\'edia & Mediana & DP & CV (\\%) "
                "& M\\'in.--M\\'ax. \\\\"),
        rows=rows,
        fonte=("Elabora\\c{c}\\~ao pr\\'opria a partir do RREO-Anexo 03 (SICONFI), "
               "deflacionado pelo IPCA (BCB/SGS 433)."),
    )


def _stl_strengths(series: pd.Series, period: int = 12) -> tuple[float, float]:
    """Forca de tendencia (F_T) e de sazonalidade (F_S), Wang et al. (2006).

    F_T = max(0, 1 - Var(R) / Var(T + R)); F_S = max(0, 1 - Var(R)/Var(S + R)),
    com T, S, R os componentes de tendencia, sazonalidade e residuo da
    decomposicao STL. Valores proximos de 1 indicam componente forte.
    """
    from statsmodels.tsa.seasonal import STL

    res = STL(series, period=period, robust=True).fit()
    r, t, s = res.resid, res.trend, res.seasonal
    var_r = float(np.var(r))
    f_t = max(0.0, 1.0 - var_r / float(np.var(t + r))) if np.var(t + r) > 0 else 0.0
    f_s = max(0.0, 1.0 - var_r / float(np.var(s + r))) if np.var(s + r) > 0 else 0.0
    return f_t, f_s


def stationarity_table(cfg: PipelineConfig, monthly_df: pd.DataFrame | None = None) -> Path:
    """Gera tab_estacionariedade.tex.

    Para cada serie (deflacionada e ja tratada), reporta os testes ADF
    (nula: raiz unitaria) e KPSS (nula: estacionariedade) em nivel, a ordem
    de diferenciacao regular ``d`` (sugerida pelo KPSS) e sazonal ``D``
    (sugerida pelo teste OCSB), e as forcas de tendencia e sazonalidade
    extraidas da decomposicao STL. A combinacao dos dois testes evita o
    diagnostico de um so lado e alimenta diretamente a busca do ``auto_arima``
    (Secao 4.5.4).
    """
    from pmdarima.arima.utils import ndiffs, nsdiffs
    from statsmodels.tsa.stattools import adfuller, kpss

    series = prepare_series(cfg, impute=True)
    rows: list[str] = []
    muns = list(cfg.municipalities.items())
    for i, (key, mun) in enumerate(muns):
        for tributo in cfg.tributos:
            s = series[(key, tributo)]
            y = s.to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_stat, adf_p = adfuller(y, autolag="AIC")[:2]
                kpss_stat, kpss_p = kpss(y, regression="c", nlags="auto")[:2]
                d = int(ndiffs(y, test="kpss", max_d=2))
                D = int(nsdiffs(y, m=12, test="ocsb", max_D=1))
            f_t, f_s = _stl_strengths(s)
            # statsmodels limita o p-valor do KPSS a [0,01; 0,10] (tabela do teste).
            kpss_txt = ("$<$0,01" if kpss_p <= 0.01
                        else "$>$0,10" if kpss_p >= 0.10 else format_brl(kpss_p, 3))
            rows.append(
                f"{mun.name} & {tributo} & {adf_stat:.2f} & {format_brl(adf_p, 3)} "
                f"& {kpss_stat:.2f} & {kpss_txt} "
                f"& {d} & {D} & {format_brl(f_t, 2)} & {format_brl(f_s, 2)} \\\\"
            )
        if i < len(muns) - 1:
            rows.append(r"\addlinespace")
    return _write_table(
        cfg,
        name="tab_estacionariedade",
        gerado_por="eda.stationarity_table",
        caption=("Testes de estacionariedade (ADF e KPSS, em n\\'ivel) e "
                 "diagn\\'ostico de diferencia\\c{c}\\~ao e for\\c{c}a dos componentes "
                 "(STL) por s\\'erie."),
        label="tab:estacionariedade",
        colspec="l l C C C C C C C C",
        size="footnotesize",
        header=("Munic\\'ipio & Tributo & ADF & $p_{\\text{ADF}}$ & KPSS & "
                "$p_{\\text{KPSS}}$ & $d$ & $D$ & $F_T$ & $F_S$ \\\\"),
        rows=rows,
        footnote=("ADF testa $H_0$: raiz unit\\'aria "
                  "(s\\'erie n\\~ao estacion\\'aria); KPSS testa $H_0$: estacionariedade. "
                  "$d$/$D$: ordens de diferencia\\c{c}\\~ao regular/sazonal sugeridas "
                  "(KPSS/OCSB), insumo do \\texttt{auto\\_arima}. $F_T$/$F_S$: for\\c{c}a "
                  "de tend\\^encia/sazonalidade (STL, Wang et al., 2006)."),
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
    )


def coverage_table(cfg: PipelineConfig, collection_report: dict | None = None) -> Path:
    """Gera tab_cobertura_dados.tex: cobertura CONJUNTA da serie mensal
    (RREO-Anexo 03) e da projecao anual da prefeitura, por (municipio, tributo).

    Consolida o que antes eram duas tabelas redundantes (ambas atestavam
    cobertura completa). Conta meses presentes/ausentes direto do
    ``monthly_revenue.csv`` e exercicios com previsao da prefeitura direto do
    ``prefeitura_forecast.csv`` -- nao depende de modelagem. A queda atipica do
    ISSQN de Camacari em 2016 nao e lacuna (os doze meses existem), e sim
    n\\'ivel deslocado, tratado no pre-processamento.
    """
    from forecasting.io import load_prefeitura_forecast

    series = prepare_series(cfg, impute=False)
    # Meses esperados = tamanho da janela amostral do cfg (jan/2015 a dez/2025 = 132),
    # derivado em vez de cravado para acompanhar cfg.sample_window automaticamente.
    expected = len(pd.period_range(cfg.sample_window.start, cfg.sample_window.end, freq="M"))
    try:
        pf = load_prefeitura_forecast(cfg)
        pf_years = pf.groupby(["cod_ibge", "tributo"])["year"].nunique().to_dict()
    except Exception:  # noqa: BLE001 -- ausencia do arquivo nao deve quebrar a EDA
        pf_years = {}
    rows: list[str] = []
    muns = list(cfg.municipalities.items())
    for i, (key, mun) in enumerate(muns):
        for tributo in cfg.tributos:
            s = series[(key, tributo)]
            present = int(s.notna().sum())
            gaps = expected - present
            anos_pf = pf_years.get((mun.cod_ibge, tributo), "--")
            obs = (r"queda at\'ipica em 2016 (\S\ref{subsec:dados-ausentes})"
                   if (key, tributo) == ("camacari", "ISSQN") else "---")
            rows.append(
                f"{mun.name} & {tributo} & {present} & {gaps} & {anos_pf} & {obs} \\\\"
            )
        if i < len(muns) - 1:
            rows.append(r"\addlinespace")
    return _write_table(
        cfg,
        name="tab_cobertura_dados",
        gerado_por="eda.coverage_table",
        caption=("Cobertura das s\\'eries, por munic\\'ipio e tributo: meses da "
                 "s\\'erie mensal coletada (RREO-Anexo 03) e exerc\\'icios com previs\\~ao "
                 "atualizada da prefeitura, no per\\'iodo de 2015 a 2025."),
        label="tab:cobertura-dados",
        # footnotesize + colunas numericas naturais (c): os cabecalhos longos
        # "Meses presentes"/"Anos com previsao" cabem em UMA linha; so a nota
        # descritiva longa em "Observacoes" (L) quebra -- em 2 linhas limpas.
        colspec="l l c c c L",
        header=("Munic\\'ipio & Tributo & Meses presentes & Lacunas & Anos com previs\\~ao "
                "& Observa\\c{c}\\~oes \\\\"),
        rows=rows,
        size="footnotesize",
        footnote=(f"Meses esperados: {expected} (jan/2015 a dez/2025). A "
                  "previs\\~ao da prefeitura prov\\'em da ``Previs\\~ao Atualizada'' do "
                  "6\\textsuperscript{o} bimestre do RREO-Anexo 03."),
        fonte="Elabora\\c{c}\\~ao pr\\'opria a partir do RREO-Anexo 03 (SICONFI).",
    )


# ---------- Figuras da EDA -----------------------------------------------

_MUN_ORDER = ["salvador", "camacari", "ilheus"]
_MUN_LABEL = {"salvador": "Salvador", "camacari": "Camaçari", "ilheus": "Ilhéus"}
_TAX_ORDER = ["IPTU", "ISSQN"]


def time_series_panel(cfg: PipelineConfig, monthly_df: pd.DataFrame | None = None) -> Path:
    """Gera fig_serie_temporal.pdf (painel 3x2: municipio x tributo).

    Mostra a serie observada (deflacionada, sem imputacao), de modo que a
    anomalia do ISSQN de Camacari em 2016 permaneca visivel.
    """
    import matplotlib.pyplot as plt

    setup_matplotlib_thesis()
    series = prepare_series(cfg, impute=False)
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 6.6), sharex=True)
    for r, key in enumerate(_MUN_ORDER):
        for c, tributo in enumerate(_TAX_ORDER):
            ax = axes[r, c]
            s = series[(key, tributo)] / 1e6
            ax.plot(s.index, s.to_numpy(), color="#45413A", lw=0.9)
            if r == 0:
                ax.set_title(tributo)
            if c == 0:
                ax.set_ylabel(f"{_MUN_LABEL[key]}\nR\\$ mi", fontsize=8)
    out = save_figure(fig, "fig_serie_temporal", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def seasonal_subseries_panel(cfg: PipelineConfig, monthly_df: pd.DataFrame | None = None) -> Path:
    """Gera fig_subseries_sazonais.pdf (painel 3x2).

    Para cada serie, sobrepoe as trajetorias anuais mes a mes (jan..dez) e a
    media de cada mes, tornando visivel a concentracao do IPTU no primeiro
    trimestre frente ao fluxo mais distribuido do ISSQN.
    """
    import matplotlib.pyplot as plt

    setup_matplotlib_thesis()
    series = prepare_series(cfg, impute=True)
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 6.6))
    for r, key in enumerate(_MUN_ORDER):
        for c, tributo in enumerate(_TAX_ORDER):
            ax = axes[r, c]
            s = series[(key, tributo)] / 1e6
            piv = pd.DataFrame({"y": s.to_numpy(), "m": s.index.month,
                                "yr": s.index.year}).pivot(index="m", columns="yr", values="y")
            for yr in piv.columns:
                ax.plot(piv.index, piv[yr].to_numpy(), color="0.7", lw=0.5)
            ax.plot(piv.index, piv.mean(axis=1).to_numpy(), color="#D55E00", lw=1.6)
            ax.set_xticks([1, 4, 7, 10])
            ax.set_xticklabels(["J", "A", "J", "O"])
            if r == 0:
                ax.set_title(tributo)
            if c == 0:
                ax.set_ylabel(f"{_MUN_LABEL[key]}\nR\\$ mi", fontsize=8)
    out = save_figure(fig, "fig_subseries_sazonais", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def stl_decomposition_panel(cfg: PipelineConfig, monthly_df: pd.DataFrame | None = None) -> Path:
    """Gera fig_stl.pdf (decomposicao STL de duas series-arquetipo).

    Decompoe o IPTU e o ISSQN de Salvador -- arquetipos da sazonalidade severa
    (imposto patrimonial concentrado no inicio do ano) e da sazonalidade suave
    (imposto de fluxo) -- em observado, tendencia, sazonalidade e residuo.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import STL

    setup_matplotlib_thesis()
    series = prepare_series(cfg, impute=True)
    cols = [("salvador", "IPTU"), ("salvador", "ISSQN")]
    comp_names = ["Observado", "Tendência", "Sazonal", "Resíduo"]
    fig, axes = plt.subplots(4, 2, figsize=(6.2, 7.0), sharex=True)
    for c, (key, tributo) in enumerate(cols):
        s = series[(key, tributo)] / 1e6
        res = STL(s, period=12, robust=True).fit()
        comps = [s, res.trend, res.seasonal, res.resid]
        for r, (name, comp) in enumerate(zip(comp_names, comps)):
            ax = axes[r, c]
            ax.plot(s.index, np.asarray(comp), color="#45413A", lw=0.8)
            if r == 0:
                ax.set_title(f"{_MUN_LABEL[key]} · {tributo}")
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    out = save_figure(fig, "fig_stl", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def acf_pacf_panel(cfg: PipelineConfig, monthly_df: pd.DataFrame | None = None) -> Path:
    """Gera fig_acf_pacf.pdf (ACF e PACF das seis series).

    Aplica uma diferenca regular e uma sazonal (lag 12) antes de estimar as
    funcoes de autocorrelacao, expondo a estrutura que orienta a escolha das
    ordens (p,q)(P,Q) do SARIMA.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    setup_matplotlib_thesis()
    series = prepare_series(cfg, impute=True)
    keys = [(k, t) for k in _MUN_ORDER for t in _TAX_ORDER]
    fig, axes = plt.subplots(6, 2, figsize=(6.3, 8.4), sharex=True)
    for r, (key, tributo) in enumerate(keys):
        s = series[(key, tributo)]
        d = s.diff().diff(12).dropna()
        # zero=False: omite a defasagem 0 (sempre 1), que comprime a escala
        plot_acf(d, ax=axes[r, 0], lags=24, title="", zero=False,
                 vlines_kwargs={"colors": "#45413A"})
        plot_pacf(d, ax=axes[r, 1], lags=24, title="", method="ywm", zero=False,
                  vlines_kwargs={"colors": "#45413A"})
        for c in (0, 1):
            for coll in axes[r, c].collections:
                coll.set_facecolor("#45413A")
                coll.set_edgecolor("#45413A")
        axes[r, 0].set_ylabel(f"{_MUN_LABEL[key][:3]}.-{tributo}")
        if r == 0:
            axes[r, 0].set_title("ACF")
            axes[r, 1].set_title("PACF")
        if r == len(keys) - 1:
            axes[r, 0].set_xlabel("Defasagem")
            axes[r, 1].set_xlabel("Defasagem")
    out = save_figure(fig, "fig_acf_pacf", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Executa todas as funcoes de geracao de artefatos da EDA.

    Regenera a tabela descritiva (sobre a serie deflacionada como coletada), a
    tabela de cobertura, a tabela de estacionariedade e as quatro figuras de
    diagnostico. Retorna a lista de paths gerados.
    """
    raw_defl = deflate_by_ipca(load_monthly_series(cfg), base_month=cfg.ipca_base_month)
    paths = [
        descriptive_stats_table(cfg, raw_defl),
        coverage_table(cfg),
        outlier_screen(cfg),
        stationarity_table(cfg),
        time_series_panel(cfg),
        stl_decomposition_panel(cfg),
        acf_pacf_panel(cfg),
    ]
    return paths
