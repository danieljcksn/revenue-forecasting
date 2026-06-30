"""Metricas de erro e ranque inter-municipal.

Gera artefatos das Secoes 5.3 e 5.4 do TCC. Escopo enxuto:
- Pontuais: MAE, MAPE, MASE
- Tabelas: metricas consolidadas e ranque dos modelos por municipio
- Figuras: boxplot do MASE por modelo (com facetas h=1 e h=12)

Decisao deliberada: ficamos com tres metricas complementares e ranking
visual, sem teste de Diebold-Mariano. Para um TCC com 6 series e poucas
dobras, diferencas de MASE > 10% ja sao narrativamente robustas; o teste
DM-HLN seria custo de defesa elevado para ganho informativo modesto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import (
    MODEL_ORDER,
    MODEL_TEX,
    PipelineConfig,
    format_dec,
    series_keys,
)

# ---------- Metricas pontuais --------------------------------------------


def _aligned(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray):
    """Converte para arrays numericos alinhados, descartando pares com NaN."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"y_true e y_pred com shapes distintos: {a.shape} vs {b.shape}")
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error: media de |y - y_hat|, na escala da serie (R$)."""
    a, b = _aligned(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error em pontos percentuais.

    MAPE = 100 * media(|y - y_hat| / |y|). Indefinido para y = 0; como a
    arrecadacao mensal das seis series e estritamente positiva (minimo
    observado da ordem de R$ 0,26 milhao), nao ha divisao por zero. Ainda
    assim, descartamos defensivamente eventuais y = 0 para nao contaminar a
    media. A metrica e reportada como complementar -- nunca decisoria --,
    pela conhecida instabilidade do MAPE em meses de arrecadacao baixa
    (Hyndman & Koehler, 2006).
    """
    a, b = _aligned(y_true, y_pred)
    nz = a != 0.0
    if nz.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((a[nz] - b[nz]) / a[nz])))


def seasonal_naive_insample_mae(train_series: pd.Series, season: int = 12) -> float:
    """MAE in-sample do Naive Sazonal no treino -- denominador do MASE.

    (1/(n-m)) * sum_{t=m+1}^{n} |y_t - y_{t-m}|, com m = season. E a escala
    que torna o erro adimensional e comparavel entre series de magnitudes
    muito diferentes (Hyndman & Koehler, 2006).
    """
    y = np.asarray(train_series, dtype=float)
    if y.size <= season:
        raise ValueError(
            f"treino com {y.size} pontos e insuficiente para escala sazonal m={season}"
        )
    diffs = np.abs(y[season:] - y[:-season])
    return float(np.mean(diffs))


def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    train_series: pd.Series,
    season: int = 12,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    Denominador = MAE do Naive Sazonal no conjunto de TREINO (in-sample),
    nao recalculado no conjunto de teste. MASE < 1 indica que o modelo supera
    o baseline sazonal; MASE = 1 empata; MASE > 1 perde para a regra trivial
    "repita o mesmo mes do ano passado".
    """
    scale = seasonal_naive_insample_mae(train_series, season=season)
    if scale == 0.0 or np.isnan(scale):
        return float("nan")
    return mae(y_true, y_pred) / scale


# ---------- Geracao de artefatos para o TCC ------------------------------


# ---------- Camada analitica (le o cache da validacao por origem movel) ---

def load_cv(cfg: PipelineConfig) -> pd.DataFrame:
    """Carrega o cache consolidado da validacao por origem movel (cv_all.csv)."""
    path = cfg.forecasts_dir / "cv_all.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nao encontrado. Rode antes: python scripts/run_pipeline.py"
        )
    cv = pd.read_csv(path, parse_dates=["origin", "train_end", "target_date"])
    cv["abs_err"] = (cv["y_true"] - cv["y_pred"]).abs()
    cv["ape"] = 100.0 * cv["abs_err"] / cv["y_true"].abs()
    cv["scaled_err"] = cv["abs_err"] / cv["insample_scale"]
    return cv


def fold_metrics(cv: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Resume MAE, MAPE e MASE (media e desvio nas dobras) por grupo."""
    g = cv.groupby(group, observed=True)
    out = g.agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        mae_sd=("abs_err", "std"),
        mape=("ape", "mean"),
        mape_sd=("ape", "std"),
        mase=("scaled_err", "mean"),
        mase_sd=("scaled_err", "std"),
        mase_med=("scaled_err", "median"),
    ).reset_index()
    return out


def metrics_table(cfg: PipelineConfig) -> Path:
    """Gera tab_metricas_comparacao.tex (comparacao consolidada).

    Para cada horizonte e cada modelo, reporta o MASE consolidado das dobras da
    origem movel, agregando as seis series. Como a distribuicao do erro e
    assimetrica a direita (poucas dobras dificeis puxam a media para cima),
    adota-se a MEDIANA como medida central, com a media entre parenteses; o IQR
    indica a dispersao. O MAPE (mediana) acompanha como metrica complementar.
    Negrito: melhor mediana do horizonte.
    """
    from forecasting.io import table_path

    cv = load_cv(cfg)

    def _iqr(x):
        return float(x.quantile(0.75) - x.quantile(0.25))

    rows_tex: list[str] = []
    for h in (1, 12):
        sub = cv[cv["step"] == h]
        g = sub.groupby("modelo")
        med = g["scaled_err"].median()
        mean = g["scaled_err"].mean()
        iqr = g["scaled_err"].apply(_iqr)
        mape_med = g["ape"].median()
        n = g.size()
        best_med_disp = format_dec(med.min(), 2)
        h_label = "um m\\^es" if h == 1 else "doze meses"
        rows_tex.append(
            f"\\multicolumn{{5}}{{l}}{{\\textit{{Horizonte $h={h}$ ({h_label})}}}} \\\\")
        for m in MODEL_ORDER:
            if m not in med.index:
                continue
            med_cell = format_dec(med[m], 2)
            extra = f"({format_dec(mean[m], 2)})"
            iqr_cell = format_dec(iqr[m], 2)
            mape_cell = format_dec(mape_med[m], 1)
            name = MODEL_TEX[m]
            # Negrito na melhor mediana do horizonte, incluindo empates ao display.
            if med_cell == best_med_disp:
                name = f"\\textbf{{{name}}}"
                med_cell = f"\\textbf{{{med_cell}}}"
            rows_tex.append(
                f"\\quad {name} & {int(n[m])} & {med_cell}~{extra} & {iqr_cell} "
                f"& {mape_cell}\\,\\% \\\\")
        if h == 1:
            rows_tex.append(r"\addlinespace")
    from forecasting.config import styled_table
    tex = styled_table(
        gerado_por="evaluation.metrics_table",
        caption=("Desempenho preditivo consolidado por horizonte: MASE "
                 "(mediana, com a m\\'edia entre par\\^enteses) e MAPE (mediana) das "
                 "dobras da origem m\\'ovel, agregando as seis s\\'eries."),
        label="tab:metricas-comparacao",
        colspec="l C C C C",
        header=["Modelo", "Dobras", "MASE (mediana)", "IQR", "MAPE"],
        rows=rows_tex,
        footnote=("MASE $<1$: desempenho superior ao \\emph{baseline} "
                  "Na\\\"ive sazonal. Mediana adotada por robustez \\`a assimetria "
                  "do erro; IQR = amplitude interquartil. Negrito: melhor mediana "
                  "do horizonte."),
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=False,
        size="footnotesize",
    )
    out = table_path(cfg, "tab_metricas_comparacao")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def metrics_by_series_table(cfg: PipelineConfig) -> Path:
    """Gera tab_metricas_por_serie.tex (Apendice): MAE (R$ mi), MAPE e MASE
    por serie, modelo e horizonte. Da o MAE em reais, dependente de escala,
    que a tabela consolidada omite."""
    from forecasting.io import table_path

    cv = load_cv(cfg)
    summ = fold_metrics(cv[cv["step"].isin([1, 12])],
                        ["municipio_nome", "tributo", "modelo", "step"])
    rows: list[str] = []
    keys = [(name, t) for _k, name, t in series_keys(cfg)]
    for i, (mun, trib) in enumerate(keys):
        block = summ[(summ["municipio_nome"] == mun) & (summ["tributo"] == trib)]
        first = True
        for m in MODEL_ORDER:
            for h in (1, 12):
                r = block[(block["modelo"] == m) & (block["step"] == h)]
                if r.empty:
                    continue
                r = r.iloc[0]
                head = f"{mun} {trib}" if first else ""
                first = False
                mae_mi = format_dec(r['mae'] / 1e6, 2)
                mape = format_dec(r['mape'], 1)
                mase = format_dec(r['mase'], 2)
                rows.append(f"{head} & {MODEL_TEX[m]} & {h} & {mae_mi} & {mape} & {mase} \\\\")
        if i < len(keys) - 1:
            rows.append(r"\addlinespace")
    body = "\n".join(rows)
    tex = (
        "% Gerado por evaluation.metrics_by_series_table -- nao editar a mao.\n"
        "\\begin{table}[htb]\n\\centering\n"
        "\\caption{M\\'etricas de erro por s\\'erie, modelo e horizonte "
        "(m\\'edia nas dobras).}\n"
        "\\label{tab:metricas-por-serie}\n\\footnotesize\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "S\\'erie & Modelo & $h$ & MAE (R\\$ mi) & MAPE (\\%) & MASE \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\fonte{Elabora\\c{c}\\~ao pr\\'opria.}\n"
        "\\end{table}\n"
    )
    out = table_path(cfg, "tab_metricas_por_serie")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def municipality_rank_table(cfg: PipelineConfig) -> Path:
    """Gera tab_ranque_municipios.tex.

    Para cada (municipio, tributo), ordena os quatro metodos pela MEDIANA do
    MASE em h=12 e reporta a sequencia 1o-4o lugar e a mediana do vencedor.
    Permite ler descritivamente se a ordenacao muda conforme o perfil do
    municipio -- sem teste formal (tres municipios sao poucos para isso).
    """
    from forecasting.io import table_path

    cv = load_cv(cfg)
    sub = cv[cv["step"] == 12]
    summ = fold_metrics(sub, ["municipio_nome", "tributo", "modelo"])
    rows: list[str] = []
    keys = [(name, t) for _k, name, t in series_keys(cfg)]
    for i, (mun, trib) in enumerate(keys):
        block = summ[(summ["municipio_nome"] == mun) &
                     (summ["tributo"] == trib)].sort_values("mase_med")
        models_tex = [MODEL_TEX[m] for m in block["modelo"]]
        if models_tex:
            # Negrito no primeiro colocado, isto e, no melhor modelo da linha.
            models_tex[0] = f"\\textbf{{{models_tex[0]}}}"
        order = " $\\succ$ ".join(models_tex)
        winner = block.iloc[0]
        wmase = format_dec(winner['mase_med'], 2)
        rows.append(f"{mun} & {trib} & {order} & {wmase} \\\\")
        if trib == "ISSQN" and i < len(keys) - 1:
            rows.append(r"\addlinespace")
    from forecasting.config import styled_table
    tex = styled_table(
        gerado_por="evaluation.municipality_rank_table",
        caption=("Ranque dos m\\'etodos pela mediana do MASE em $h=12$, por "
                 "munic\\'ipio e tributo (ordem decrescente de desempenho)."),
        label="tab:ranque-municipios",
        colspec="l l L r",
        header=["Munic\\'ipio", "Tributo",
                "Ranque ($1^{\\text{o}} \\succ 4^{\\text{o}}$)",
                "MASE do $1^{\\text{o}}$"],
        rows=rows,
        footnote=("$\\succ$ indica desempenho superior (menor MASE); em negrito, "
                  "o 1\\textsuperscript{o} colocado. "
                  "MASE $<1$: supera o \\emph{baseline} Na\\\"ive sazonal."),
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=True,
        size="small",
    )
    out = table_path(cfg, "tab_ranque_municipios")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def mase_boxplot(cfg: PipelineConfig) -> Path:
    """Gera fig_boxplot_mase.pdf: distribuicao do MASE por modelo, facetada por
    horizonte (h=1 e h=12). Cada ponto e uma dobra; a linha em MASE=1 marca o
    baseline. Mostra dispersao e estabilidade, nao so a media."""
    import matplotlib.pyplot as plt

    from forecasting.plotting import model_boxplot, save_figure, setup_matplotlib_thesis

    setup_matplotlib_thesis()
    cv = load_cv(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.3), sharey=True)
    for ax, h in zip(axes, (1, 12)):
        data = [cv[(cv["step"] == h) & (cv["modelo"] == m)]["scaled_err"].to_numpy()
                for m in MODEL_ORDER]
        model_boxplot(ax, data, MODEL_ORDER)
        ax.set_title(f"Horizonte $h = {h}$")
        if h == 1:
            ax.set_ylabel("MASE (por dobra)")
    out = save_figure(fig, "fig_boxplot_mase", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def mase_heatmap(cfg: PipelineConfig) -> Path:
    """Gera fig_mase_heatmap.pdf: mapa de calor do MASE mediano em h=12.

    Linhas = as seis series (3 IPTU em cima, 3 ISSQN embaixo, com divisor e
    rotulos de bloco); colunas = os seis modelos na ordem canonica. Cor
    sequencial quente da casa (creme p/ MASE baixo/bom -> tinta escura p/ alto/
    ruim). Cada celula traz o valor (vircula, 2 casas); o menor de cada linha
    (vencedor) sai em negrito com contorno. Le-se de relance: bloco IPTU claro
    (baixo) vs bloco ISSQN escuro (alto). So le o cache e plota."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    from forecasting.config import MODEL_LABELS, format_dec
    from forecasting.plotting import save_figure, setup_matplotlib_thesis

    setup_matplotlib_thesis()
    cv = load_cv(cfg)
    fm = fold_metrics(cv[cv["step"] == 12], ["municipio_nome", "tributo", "modelo"])

    # Ordem das linhas: IPTU (3 municipios) em cima, ISSQN embaixo. Municipios
    # na ordem canonica do cfg (Salvador, Camacari, Ilheus).
    mun_order = [m.name for m in cfg.municipalities.values()]
    row_keys = [(t, mun) for t in ("IPTU", "ISSQN") for mun in mun_order]
    lookup = {(r["tributo"], r["municipio_nome"], r["modelo"]): r["mase_med"]
              for _i, r in fm.iterrows()}
    import numpy as np
    M = np.array([[lookup[(t, mun, m)] for m in MODEL_ORDER]
                  for (t, mun) in row_keys])
    n_rows, n_cols = M.shape

    # Mapa sequencial quente da casa: creme (MASE baixo/bom) -> tinta escura
    # (alto/ruim), derivado do bege do documento.
    cmap = LinearSegmentedColormap.from_list(
        "tcc_warm", ["#F4EFE4", "#D8C39A", "#B98C4E", "#8A5A2B", "#45413A"])
    norm = Normalize(vmin=0.2, vmax=1.7)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto",
              extent=(0, n_cols, n_rows, 0), interpolation="nearest")

    # Anotacao das celulas + realce do vencedor (menor MASE) de cada linha.
    for i in range(n_rows):
        j_best = int(np.argmin(M[i]))
        for j in range(n_cols):
            val = M[i, j]
            # Texto claro sobre celula escura, escuro sobre clara.
            txt_color = "#FBF8F1" if norm(val) > 0.55 else "#45413A"
            weight = "bold" if j == j_best else "regular"
            ax.text(j + 0.5, i + 0.5, format_dec(val, 2),
                    ha="center", va="center", color=txt_color,
                    fontsize=8.3, fontweight=weight)
        # Contorno no vencedor da linha.
        ax.add_patch(Rectangle((j_best, i), 1, 1, fill=False,
                               edgecolor="#FBF8F1", linewidth=1.6, zorder=4))

    # Divisor sutil entre o bloco IPTU (linhas 0-2) e ISSQN (linhas 3-5).
    ax.axhline(3, color="#FBF8F1", linewidth=2.4, zorder=5)

    # Rotulos de coluna (modelos) no topo.
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0.0, labelcolor="#45413A", pad=4)

    # Rotulos de linha = nome do municipio.
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([mun for (_t, mun) in row_keys])
    ax.tick_params(axis="y", length=0.0, labelcolor="#45413A", pad=4)

    # Rotulos de bloco (IPTU / ISSQN) a esquerda, fora do eixo (afastados dos
    # rotulos de municipio para nao colidir com descendentes tipo "Camaçari").
    ax.text(-0.115, 0.815, "IPTU", transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#45413A")
    ax.text(-0.115, 0.27, "ISSQN", transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#45413A")

    # Limpa molduras e grade (o proprio mapa ja delimita).
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)

    out = save_figure(fig, "fig_mase_heatmap", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def horizonte_curva(cfg: PipelineConfig) -> Path:
    """Gera fig_horizonte_curva.pdf: MASE mediano de cada modelo ao longo do
    horizonte (h = 1..12).

    Uma linha por modelo (cor canonica), com o Naive sazonal em destaque como
    referencia e uma linha tenue em MASE=1. Torna visivel o cruzamento: ETS/
    Theta partem baixos em h=1 e sobem; o Naive alcanca e ultrapassa varios no
    horizonte longo (melhor mediana em h=12). So le o cache e plota."""
    import matplotlib.pyplot as plt

    from forecasting.config import MODEL_COLORS, MODEL_LABELS
    from forecasting.plotting import (
        BASELINE_GREY,
        clean_legend,
        save_figure,
        setup_matplotlib_thesis,
        style_axis,
    )

    setup_matplotlib_thesis()
    cv = load_cv(cfg)
    med = cv.groupby(["step", "modelo"])["scaled_err"].median().unstack()
    steps = med.index.to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    # Referencia em MASE = 1 (baseline sazonal empata). Rotulo logo acima da
    # linha, recuado da borda direita para nao encostar na moldura.
    ax.axhline(1.0, color=BASELINE_GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.text(steps[-1], 1.015, "MASE = 1", color="#6E695E",
            fontsize=7.8, va="bottom", ha="right")

    # Demais modelos (linha fina, leve transparencia) e o Naive em destaque.
    for m in MODEL_ORDER:
        if m not in med.columns:
            continue
        y = med[m].to_numpy()
        if m == "Naive":
            continue
        ax.plot(steps, y, color=MODEL_COLORS[m], lw=1.6, alpha=0.95,
                label=MODEL_LABELS[m], zorder=3)

    # Naive sazonal por ultimo, em destaque (linha mais grossa + marcadores).
    yn = med["Naive"].to_numpy()
    ax.plot(steps, yn, color=MODEL_COLORS["Naive"], lw=2.8,
            marker="o", markersize=4.0, markerfacecolor=MODEL_COLORS["Naive"],
            markeredgecolor="white", markeredgewidth=0.8,
            label=MODEL_LABELS["Naive"], zorder=5)

    ax.set_xlabel("Horizonte de previsão (meses)")
    ax.set_ylabel("MASE (mediano por dobra)")
    ax.set_xticks(steps)
    ax.set_xlim(steps[0] - 0.3, steps[-1] + 0.3)
    style_axis(ax)

    # Legenda enxuta, horizontal, acima do conjunto (ordem do MODEL_ORDER).
    handles, labels = ax.get_legend_handles_labels()
    order = {MODEL_LABELS[m]: i for i, m in enumerate(MODEL_ORDER)}
    pairs = sorted(zip(handles, labels), key=lambda hl: order.get(hl[1], 99))
    handles, labels = zip(*pairs)
    clean_legend(fig, list(handles), list(labels), ncol=6)

    out = save_figure(fig, "fig_horizonte_curva", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def covid_regime_note(cfg: PipelineConfig) -> Path:
    """Materializa o efeito do regime pandemico, cumprindo a promessa
    metodologica de reportar metricas separadas por regime temporal.

    Calcula o MASE medio das dobras cuja janela de teste cai na cauda da
    pandemia (2021) versus o regime de normalizacao (2022--2025), por modelo
    e horizonte, e grava em ``data/forecasts/covid_regime.txt``. Da
    proveniencia auditavel aos numeros citados na nota sobre o periodo
    pandemico (Secao 5 dos resultados).
    """
    from forecasting.models import covid_regime

    cv = load_cv(cfg).copy()
    cv["scaled_err"] = (cv["y_true"] - cv["y_pred"]).abs() / cv["insample_scale"]
    # Mesmo corte de regime que run_pipeline (models.covid_regime); como
    # covid_period.end=2021-12-31, {pre,covid} == ano<=2021, preservando os
    # numeros da nota (pandemia 2021 vs normalizacao 2022-2025).
    _reg = pd.to_datetime(cv["target_date"]).apply(lambda d: covid_regime(d, cfg))
    cv["regime"] = np.where(_reg.isin(["pre", "covid"]), "pandemia_2021", "pos_2022_2025")
    g = (cv.groupby(["step", "modelo", "regime"])["scaled_err"]
         .mean().reset_index().sort_values(["step", "modelo", "regime"]))
    lines = ["# MASE medio por regime temporal: cauda da pandemia (2021) vs",
             "# normalizacao (2022-2025), por horizonte e modelo.",
             "# step : modelo : regime : MASE_medio"]
    for _, r in g.iterrows():
        lines.append(f"{int(r['step'])}: {r['modelo']}: {r['regime']}: "
                     f"{r['scaled_err']:.3f}")
    out = cfg.forecasts_dir / "covid_regime.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Gera os artefatos de avaliacao (Secoes 5.3 e 5.4) a partir do cache."""
    # metrics_by_series_table() NAO entra no build: a tabela por serie nao e
    # usada no corpo (seu conteudo ja esta em tab_ranque_municipios +
    # tab_metricas_comparacao). A funcao fica disponivel para um eventual
    # apendice, mas nao se gera artefato morto por padrao.
    return [
        metrics_table(cfg),
        municipality_rank_table(cfg),
        mase_boxplot(cfg),
        covid_regime_note(cfg),
    ]
