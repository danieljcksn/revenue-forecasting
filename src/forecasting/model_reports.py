"""Tabelas de parametros e figuras por modelo (Secao 5.2 do TCC).

Le o cache da validacao por origem movel (cv_all.csv) para as figuras de
previsao versus realizado e ajusta cada modelo UMA vez na serie completa
(barato) para as tabelas de parametros e os diagnosticos. Reflete o PORTFOLIO
NOVO da reauditoria: ETS via statsforecast AutoETS (taxonomia completa), SARIMA
em statsmodels com D=1 (ordens da janela inicial), Prophet mensal corrigido.
Nao re-executa a validacao por origem movel.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    PipelineConfig,
    format_brl,
    format_dec,
    series_keys,
)
from forecasting.plotting import setup_matplotlib_thesis

warnings.filterwarnings("ignore")

_MUN_ORDER = ["salvador", "camacari", "ilheus"]
_MUN_LABEL = {"salvador": "Salvador", "camacari": "Camaçari", "ilheus": "Ilhéus"}
_TAX_ORDER = ["IPTU", "ISSQN"]

# Ordens SARIMA com D=1, selecionadas pelo auto_arima (test=KPSS, D forcado=1)
# sobre a AMOSTRA COMPLETA -- coerente com o "ajuste na amostra completa" da
# tabela (a validacao por origem movel congela as ordens da janela inicial, sem
# vazamento; convencao identica a do monografia original). Fixadas aqui para nao
# reintroduzir a dependencia de pmdarima no ambiente de build (statsforecast/numba
# usa numpy 2.x, incompativel com o pmdarima compilado).
_SARIMA_D1_ORDERS = {
    ("salvador", "IPTU"):  ((1, 0, 0), (0, 1, 1, 12)),
    ("salvador", "ISSQN"): ((1, 1, 2), (0, 1, 1, 12)),
    ("camacari", "IPTU"):  ((0, 0, 0), (0, 1, 1, 12)),
    ("camacari", "ISSQN"): ((2, 1, 0), (0, 1, 1, 12)),
    ("ilheus",   "IPTU"):  ((0, 0, 1), (0, 1, 1, 12)),
    ("ilheus",   "ISSQN"): ((1, 1, 1), (0, 1, 1, 12)),
}


# ---------- Tabelas de parametros ----------------------------------------


def ets_params_table(cfg: PipelineConfig) -> Path:
    """tab_ets_params.tex: especificacao ETS selecionada pela AutoETS (taxonomia
    completa, erro aditivo OU multiplicativo) e parametros de suavizacao
    (alpha, beta, gamma, phi) e AICc, por serie."""
    from statsforecast.models import AutoETS

    from forecasting.config import styled_table
    from forecasting.eda import prepare_series
    from forecasting.io import table_path

    series = prepare_series(cfg, impute=True)
    keys = series_keys(cfg)
    rows: list[str] = []
    for i, (mk, name, trib) in enumerate(keys):
        y = np.asarray(series[(mk, trib)], dtype=float)
        m = AutoETS(season_length=12, model="ZZZ").fit(y)
        md = m.model_
        spec = str(md["method"])  # ex.: "ETS(M,A,A)" / "ETS(A,Ad,M)"
        par = np.asarray(md["par"], dtype=float)
        # statsforecast reserva [alpha, beta, gamma, phi] nas 4 primeiras
        # posicoes; componente ausente vem como NaN (-> "--" no format_dec).
        alpha, beta, gamma, phi = par[0], par[1], par[2], par[3]
        aicc = float(md.get("aicc", float("nan")))
        rows.append(
            f"{name} & {trib} & {spec} & {format_dec(alpha)} & "
            f"{format_dec(beta)} & {format_dec(gamma)} & {format_dec(phi)} & "
            f"{format_brl(aicc, 1)} \\\\")
        if trib == "ISSQN" and i < len(keys) - 1:
            rows.append(r"\addlinespace")
    tex = styled_table(
        gerado_por="model_reports.ets_params_table",
        caption="Especifica\\c{c}\\~oes ETS por s\\'erie",
        label="tab:ets-params",
        colspec="l l l C C C C C",
        header=["Munic\\'ipio", "Tributo", "Especifica\\c{c}\\~ao", "$\\alpha$",
                "$\\beta$", "$\\gamma$", "$\\phi$", "AICc"],
        rows=rows,
        # Nota alinhada ao texto defendido (a versao commitada do artefato foi
        # hand-editada em 02/07 sem retroalimentar o gerador; aqui reconcilia).
        footnote="Na nota\\c{c}\\~ao ETS(erro, tend\\^encia, sazonalidade), "
        "A indica componente aditivo, M multiplicativo e N ausente. "
        "A sele\\c{c}\\~ao autom\\'atica ret\\'em a configura\\c{c}\\~ao com "
        "melhor ajuste penalizado.",
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=True,
        size="footnotesize",
        floating=False,
    )
    out = table_path(cfg, "tab_ets_params")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def sarima_params_table(cfg: PipelineConfig) -> Path:
    """tab_sarima_params.tex: ordens (p,d,q)(P,D,Q)_12 com D=1, AIC e p-valor de
    Ljung-Box dos residuos, por serie (ajuste statsmodels na amostra completa)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    from forecasting.config import styled_table
    from forecasting.eda import prepare_series
    from forecasting.io import table_path
    from forecasting.models import fit_sarimax_fixed

    series = prepare_series(cfg, impute=True)
    keys = series_keys(cfg)
    rows: list[str] = []
    for i, (mk, name, trib) in enumerate(keys):
        order, sorder = _SARIMA_D1_ORDERS[(mk, trib)]
        fm = fit_sarimax_fixed(series[(mk, trib)], order, sorder, use_log=True)
        model = fm.fit_object
        p, d, q = order
        P, D, Q = sorder[:3]
        resid = np.asarray(model.resid)
        try:
            lb = acorr_ljungbox(resid, lags=[2 * 12], model_df=p + q + P + Q,
                                return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            lb_p = float("nan")
        order_tex = f"$({p},{d},{q})({P},{D},{Q})_{{12}}$"
        aic_txt = format_dec(fm.aic, 1).replace("-", "$-$")
        rows.append(
            f"{name} & {trib} & {order_tex} & {aic_txt} & "
            f"{format_dec(lb_p, 3)} \\\\")
        if trib == "ISSQN" and i < len(keys) - 1:
            rows.append(r"\addlinespace")
    tex = styled_table(
        gerado_por="model_reports.sarima_params_table",
        caption="Especifica\\c{c}\\~oes SARIMA por s\\'erie",
        label="tab:sarima-params",
        colspec="l l L C C",
        header=["Munic\\'ipio", "Tributo", "Especifica\\c{c}\\~ao", "AIC",
                "$p$ (Ljung-Box)"],
        rows=rows,
        footnote="$D=1$ for\\c{c}ado (coerente com a forte sazonalidade $F_S$ da "
        "EDA; o OCSB subdiagnosticava $D$). $p$ (Ljung-Box, defasagem $2s=24$, "
        "g.l. ajustados por $p+q+P+Q$): nas s\\'eries de ISSQN $p>0{,}05$ "
        "(res\\'iduos compat\\'iveis com ru\\'ido branco); no IPTU a "
        "especifica\\c{c}\\~ao parcimoniosa com $D=1$ prioriza a estabilidade e a "
        "acur\\'acia do agregado anual (sem a explos\\~ao de longo horizonte do "
        "ajuste $d{=}1,D{=}0$) e rejeita a hip\\'otese de ru\\'ido branco, "
        "limita\\c{c}\\~ao reportada com transpar\\^encia.",
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=True,
        size="footnotesize",
        floating=False,
    )
    out = table_path(cfg, "tab_sarima_params")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


# Estilo por modelo (nome, rotulo de exibicao, cor canonica), da FONTE UNICA
# (config.MODEL_ORDER/MODEL_LABELS/MODEL_COLORS). Sem hex solto.
_MODEL_STYLE = [(m, MODEL_LABELS[m], MODEL_COLORS[m]) for m in MODEL_ORDER]


def forecasts_consolidated_figure(cfg: PipelineConfig) -> Path:
    """fig_forecasts_formais.pdf: painel 3x2 (municipio x tributo) com a serie
    realizada, a AMPLITUDE das previsoes um-passo-a-frente dos seis modelos
    (faixa min-max, sem espaguete de linhas) e o Ensemble destacado como
    recomendacao operacional. Le apenas o cache (h=1)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from forecasting.eda import prepare_series
    from forecasting.evaluation import load_cv
    from forecasting.plotting import REALIZED_INK, save_figure, style_axis

    setup_matplotlib_thesis()
    cv = load_cv(cfg)
    series = prepare_series(cfg, impute=True)
    first_origin = pd.to_datetime(cv["origin"]).min()
    band_fill = "#D3E6FB"      # azul-superficie: faixa entre os seis previsores
    ens_color = MODEL_COLORS["Ensemble"]
    fig, axes = plt.subplots(3, 2, figsize=(6.0, 6.4), sharex=True)
    for r, mk in enumerate(_MUN_ORDER):
        for c, trib in enumerate(_TAX_ORDER):
            ax = axes[r, c]
            s = series[(mk, trib)] / 1e6
            sub = cv[(cv["municipio"] == mk) & (cv["tributo"] == trib) &
                     (cv["step"] == 1)]
            wide = (sub.pivot_table(index="target_date", columns="modelo",
                                    values="y_pred").sort_index() / 1e6)
            ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1),
                            color=band_fill, lw=0, zorder=1)
            ax.plot(wide.index, wide["Ensemble"], color=ens_color, lw=1.05,
                    zorder=3)
            ax.plot(s.index, s.to_numpy(), color=REALIZED_INK, lw=1.15, zorder=4)
            ax.axvline(first_origin, color="#B7BCC2", lw=0.8, ls=(0, (3, 3)),
                       zorder=2)
            if r == 0 and c == 0:
                ax.annotate("início das\nprevisões", xy=(first_origin, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(-4, -2), textcoords="offset points",
                            ha="right", va="top", fontsize=7.0, color="#5F6368")
            ax.margins(y=0.08)
            style_axis(ax)
            if r == 0:
                ax.set_title(trib)
            if c == 0:
                ax.set_ylabel(f"{_MUN_LABEL[mk]}\n(R\\$ milhões)", fontsize=8.0)
    fig.align_ylabels(axes[:, 0])
    handles = [
        Line2D([0], [0], color=REALIZED_INK, lw=1.5, label="Realizado"),
        Patch(facecolor=band_fill, label="Faixa entre os modelos"),
        Line2D([0], [0], color=ens_color, lw=1.5, label="Ensemble"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=7.7)
    out = save_figure(fig, "fig_forecasts_formais", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Gera as tabelas de parametros e a figura de previsoes (Secao 5.2)."""
    return [
        ets_params_table(cfg),
        sarima_params_table(cfg),
        forecasts_consolidated_figure(cfg),
    ]
