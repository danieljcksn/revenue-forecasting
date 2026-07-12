"""Metricas de erro e ranque inter-municipal.

Gera artefatos das Secoes 5.3 e 5.4 e da Discussao do TCC. Escopo enxuto:
- Pontuais: MAE, MAPE, MASE
- Tabela: tab_metricas_comparacao (desempenho consolidado por horizonte)
- Figuras: fig_mase_heatmap, fig_horizonte_curva, fig_naive_por_ano
- Nota de proveniencia: covid_regime.txt

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
            f"{path} nao encontrado (cache canonico versionado; ver RUN_ORDER.md)"
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
            # Negrito SO na melhor mediana do horizonte (o valor ranqueado, fiel
            # a Nota); o nome do modelo permanece em peso regular, sem excesso.
            if med_cell == best_med_disp:
                med_cell = f"\\textbf{{{med_cell}}}"
            rows_tex.append(
                f"\\quad {name} & {int(n[m])} & {med_cell}~{extra} & {iqr_cell} "
                f"& {mape_cell} \\\\")
        if h == 1:
            rows_tex.append(r"\addlinespace")
    from forecasting.config import styled_table
    tex = styled_table(
        gerado_por="evaluation.metrics_table",
        caption="Desempenho consolidado por horizonte",
        label="tab:metricas-comparacao",
        colspec="l C C C C",
        header=["Modelo", "Dobras", "MASE", "IQR", "MAPE (\\%)"],
        rows=rows_tex,
        footnote=("Desempenho agregado das seis s\\'eries do n\\'ucleo (tr\\^es "
                  "munic\\'ipios $\\times$ dois tributos), consolidando todas as "
                  "dobras da valida\\c{c}\\~ao por origem m\\'ovel de cada horizonte "
                  "(coluna Dobras). Nas c\\'elulas de MASE, a mediana das dobras, "
                  "com a m\\'edia entre par\\^enteses; IQR = amplitude interquartil. "
                  "Negrito: melhor mediana do horizonte."),
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=False,
        size="footnotesize",
    )
    out = table_path(cfg, "tab_metricas_comparacao")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def mase_heatmap(cfg: PipelineConfig) -> Path:
    """Gera fig_mase_heatmap.pdf: mapa de calor do MASE mediano em h=12.

    Linhas = as seis series (3 IPTU em cima, 3 ISSQN embaixo, com divisor e
    rotulos de bloco); colunas = os seis modelos na ordem canonica. Cor
    sequencial azul da casa (quase-branco p/ MASE baixo -> azul-escuro p/
    alto). Cada celula traz o valor (virgula, 2 casas); o menor de cada linha
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
    M = np.array([[lookup[(t, mun, m)] for m in MODEL_ORDER]
                  for (t, mun) in row_keys])
    n_rows, n_cols = M.shape

    cmap = LinearSegmentedColormap.from_list(
        "munitax_mase", ["#F7FBFF", "#D8ECFF", "#8AB4F8", "#0582FF", "#174EA6"])
    norm = Normalize(vmin=0.2, vmax=1.7)

    left_cols = 2
    fig, ax = plt.subplots(figsize=(6.0, 3.15))
    ax.set_xlim(0, left_cols + n_cols)
    ax.set_ylim(n_rows + 1, 0)
    ax.set_facecolor("white")

    header_fill = "#F6F9FC"
    stripe_fill = "#FAFBFD"
    rule = "#DADCE0"
    text = "#202124"
    muted = "#5F6368"

    for j in range(left_cols + n_cols):
        ax.add_patch(Rectangle((j, 0), 1, 1, facecolor=header_fill,
                               edgecolor="white", linewidth=1.0, zorder=1))
    ax.text(0.5, 0.5, "Tributo", ha="center", va="center",
            fontsize=8.0, fontweight="semibold", color=text)
    ax.text(1.5, 0.5, "Município", ha="center", va="center",
            fontsize=8.0, fontweight="semibold", color=text)
    for j, m in enumerate(MODEL_ORDER):
        ax.text(left_cols + j + 0.5, 0.5, MODEL_LABELS[m], ha="center",
                va="center", fontsize=8.0, fontweight="semibold", color=text)

    for i, (trib, mun) in enumerate(row_keys):
        y = i + 1
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y), left_cols + n_cols, 1,
                                   facecolor=stripe_fill, edgecolor="none", zorder=0))
        ax.text(0.5, y + 0.5, trib, ha="center", va="center",
                fontsize=7.8, color=muted, fontweight="semibold")
        ax.text(1.5, y + 0.5, mun, ha="center", va="center",
                fontsize=7.8, color=text)

    for i in range(n_rows):
        j_best = int(np.argmin(M[i]))
        for j in range(n_cols):
            val = M[i, j]
            x = left_cols + j
            y = i + 1
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=cmap(norm(val)),
                                   edgecolor="white", linewidth=1.0, zorder=2))
            txt_color = "white" if norm(val) > 0.62 else text
            weight = "bold" if j == j_best else "regular"
            ax.text(x + 0.5, y + 0.5, format_dec(val, 2),
                    ha="center", va="center", color=txt_color,
                    fontsize=7.8, fontweight=weight, zorder=3)
        ax.add_patch(Rectangle((left_cols + j_best, i + 1), 1, 1, fill=False,
                               edgecolor="white", linewidth=1.8, zorder=4))

    ax.axhline(1, color=rule, linewidth=0.7, zorder=5)
    ax.axhline(4, color=rule, linewidth=0.8, zorder=5)
    ax.axvline(left_cols, color=rule, linewidth=0.7, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

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

    fig, ax = plt.subplots(figsize=(6.0, 3.05))

    # Referencia em MASE = 1 (baseline sazonal empata). Rotulo a ESQUERDA,
    # regiao livre de curvas (a direita o ETS cruza a linha em h=11).
    ax.axhline(1.0, color=BASELINE_GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.text(steps[0], 1.012, "MASE = 1", color="#5F6368",
            fontsize=7.6, va="bottom", ha="left")

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
    from forecasting.plotting import br_axis
    br_axis(ax, "y", decimals=1, step=0.1)
    # Limite superior justo ao dado (pico ETS ~1,03): elimina a faixa morta no
    # topo entre a legenda e as curvas, sem cortar nenhuma linha.
    ax.set_ylim(0.57, 1.07)
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


def naive_by_year_figure(cfg: PipelineConfig) -> Path:
    """Gera fig_naive_por_ano.pdf: MAE de cada modelo RELATIVO ao Naive, por
    ano-alvo, em h=12.

    Mostra QUANDO o baseline vence: nos exercicios de recomposicao pos-pandemia
    (2022 a 2024) o valor do mesmo mes do ano anterior acompanhou o degrau de
    nivel mais depressa que os modelos; em 2021 e 2025 os modelos ganham com
    folga. Abaixo de 1, o modelo erra menos que o Naive. Cache-only."""
    import matplotlib.pyplot as plt

    from forecasting.config import MODEL_COLORS, MODEL_LABELS
    from forecasting.plotting import (
        BASELINE_GREY,
        br_axis,
        clean_legend,
        save_figure,
        setup_matplotlib_thesis,
        style_axis,
    )

    setup_matplotlib_thesis()
    cv = load_cv(cfg)
    h12 = cv[cv["step"] == 12].copy()
    h12["ano"] = pd.to_datetime(h12["target_date"]).dt.year
    mae_ano = h12.groupby(["ano", "modelo"])["abs_err"].mean().unstack()
    rel = mae_ano.div(mae_ano["Naive"], axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.axhline(1.0, color=BASELINE_GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
    # Rotulo acima do ponto de 2021 (Theta ~1,01), com folga clara, para nao
    # colidir com o marcador ciano nem com a linha ascendente para 2022.
    ax.text(rel.index[0] - 0.34, 1.08, "Naïve = 1", color="#5F6368",
            fontsize=7.6, va="bottom", ha="left")
    for m in MODEL_ORDER:
        if m == "Naive" or m not in rel.columns:
            continue
        ax.plot(rel.index, rel[m].to_numpy(), color=MODEL_COLORS[m], lw=1.6,
                marker="o", markersize=3.6, markeredgecolor="white",
                markeredgewidth=0.7, label=MODEL_LABELS[m], zorder=3)
    ax.set_xticks(list(rel.index))
    ax.set_xlabel("Ano-alvo da previsão")
    ax.set_ylabel("MAE relativo ao Naïve")
    ax.margins(y=0.10)
    br_axis(ax, "y", decimals=1, step=0.2)
    style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    clean_legend(fig, handles, labels, ncol=5)
    out = save_figure(fig, "fig_naive_por_ano", cfg.figures_dir_abs)
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
    """Gera os artefatos de avaliacao do documento a partir do cache."""
    return [
        metrics_table(cfg),
        mase_heatmap(cfg),
        horizonte_curva(cfg),
        naive_by_year_figure(cfg),
        covid_regime_note(cfg),
    ]
