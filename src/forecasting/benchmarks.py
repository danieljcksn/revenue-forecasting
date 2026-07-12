"""Confronto com a previsao da prefeitura e com Oliveira (2024).

Gera artefatos da Secao 5.4 do TCC. Escopo enxuto: as duas comparacoes
sao executadas no mesmo notebook (04_evaluation.ipynb).

Decisoes:
- Tributos: somente IPTU e ISSQN (a comparacao com Oliveira so se aplica
  a esses dois tributos; o ITBI esta fora do escopo deste TCC).
- Janela COVID: dobras com janela de teste inteiramente em 2020-03 a
  2021-12 sao reportadas como nota de rodape. Nao ha analise de
  sensibilidade com/sem dummy estrutural (custo de defesa elevado para
  ganho marginal).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.config import (
    MODEL_ORDER,
    MODEL_TEX,
    PipelineConfig,
    format_dec,
    mun_label,
    series_keys,
)

# ---------- Numeros fixos extraidos de Oliveira (2024) ------------------
# Erro percentual da projecao da PROPRIA PREFEITURA para 2022, por (municipio,
# tributo), conforme Oliveira (2024) -- conferido contra a Tabela 01 (serie
# historica de receitas realizadas x estimadas pelos municipios) e a Tabela 10
# (consolidacao dos indices de erro). Series deflacionadas pelo IGP-M naquele
# estudo; aqui usamos IPCA -- a diferenca nao afeta razoes percentuais dentro
# de um mesmo ano. Estes nao sao os erros dos modelos de Oliveira, e sim os
# erros das previsoes que os municipios efetivamente publicaram.
OLIVEIRA_2022_ERRORS_PCT: dict[tuple[str, str], float] = {
    ("salvador", "IPTU"):   4.33,
    ("salvador", "ISSQN"):  1.04,
    ("camacari", "IPTU"):  25.49,
    ("camacari", "ISSQN"): 29.68,
    ("ilheus",   "IPTU"):   0.73,
    ("ilheus",   "ISSQN"): 28.43,
}

# Erro percentual ABSOLUTO da previsao de 2022 de cada METODO de Oliveira
# (2024), por (municipio, tributo): Box-Jenkins, Alisamento Exponencial de
# Holt-Winters e Redes Neurais Artificiais (NNAR). Valores transcritos das
# Tabelas 03-08 (individuais) e consolidados nas Tabelas 09 (ISSQN) e 10
# (IPTU) daquele trabalho; sinais negativos convertidos para magnitude, por
# comparabilidade com o erro absoluto deste estudo.
OLIVEIRA_2022_METHOD_ERRORS_PCT: dict[tuple[str, str], dict[str, float]] = {
    ("salvador", "IPTU"):  {"bj":  4.05, "hw":  0.95, "nnar":  8.88},
    ("salvador", "ISSQN"): {"bj": 11.27, "hw": 11.74, "nnar":  8.67},
    ("camacari", "IPTU"):  {"bj": 25.93, "hw": 26.52, "nnar": 28.72},
    ("camacari", "ISSQN"): {"bj":  2.58, "hw": 13.06, "nnar": 16.32},
    ("ilheus",   "IPTU"):  {"bj":  8.95, "hw":  5.22, "nnar":  7.07},
    ("ilheus",   "ISSQN"): {"bj": 25.39, "hw": 32.05, "nnar":  9.46},
}


# ---------- Benchmark da prefeitura --------------------------------------


def _sarima_jensen_annual(cfg: PipelineConfig) -> pd.DataFrame | None:
    """Agregado anual do SARIMA com correcao log-normal (Jensen): em vez de somar
    as medianas mensais exp(mu_t), soma os valores esperados exp(mu_t+sigma_t^2/2)
    (E[soma]=soma E[y_t]). Le ``data/forecasts/sarima_var.csv`` (mu_t e sigma_t por
    origem/passo, escala log; produzido pelo estagio classic de scripts/run_pipeline_full.py). Retorna
    None se o cache de variancia nao existir (graceful: cai na soma-de-medianas).

    O ponto-previsao MENSAL do SARIMA segue a mediana (otimo sob MAE/MASE); esta
    correcao incide SO no agregado anual e SO no SARIMA (unico ajustado em log)."""
    path = cfg.forecasts_dir / "sarima_var.csv"
    if not path.exists():
        return None
    var = pd.read_csv(path, parse_dates=["origin"])
    dec = var[(var["origin"].dt.month == 12) & (var["step"].between(1, 12))].copy()
    dec["target_year"] = dec["origin"].dt.year + 1
    g = dec.groupby(["municipio", "tributo", "target_year"]).agg(
        pred_annual=("y_pred_mean", "sum"), n_steps=("step", "count")).reset_index()
    return g[g["n_steps"] == 12][["municipio", "tributo", "target_year", "pred_annual"]]


def aggregate_monthly_to_annual(cv: pd.DataFrame,
                                cfg: PipelineConfig | None = None) -> pd.DataFrame:
    """Agrega as previsoes mensais em previsao anual, ano a ano.

    Seleciona as origens que terminam em dezembro (o gestor projeta, no fim de
    um exercicio, o exercicio seguinte) e soma os doze passos mensais ($h=1$ a
    $h=12$) para obter a previsao do ano calendario completo. Retorna, por
    (municipio, tributo, modelo, ano-alvo), a previsao anual, o realizado anual
    (soma dos doze meses observados) e o erro percentual do modelo.

    Quando ``cfg`` e dado e ha cache de variancia, o agregado do SARIMA usa a
    correcao de Jensen (ver ``_sarima_jensen_annual``); os demais modelos somam o
    ponto-previsao (media para ETS/Theta/Prophet/Ensemble na escala original,
    mediana para o Naive)."""
    dec = cv[(cv["origin"].dt.month == 12) & (cv["step"].between(1, 12))].copy()
    dec["target_year"] = dec["origin"].dt.year + 1
    grp = dec.groupby(["municipio", "municipio_nome", "tributo", "modelo", "target_year"])
    annual = grp.agg(pred_annual=("y_pred", "sum"),
                     real_annual=("y_true", "sum"),
                     n_steps=("step", "count")).reset_index()
    annual = annual[annual["n_steps"] == 12]  # apenas anos completos
    if cfg is not None:
        jen = _sarima_jensen_annual(cfg)
        if jen is not None:
            annual = annual.merge(jen, on=["municipio", "tributo", "target_year"],
                                  how="left", suffixes=("", "_jensen"))
            is_sarima = (annual["modelo"] == "SARIMA") & annual["pred_annual_jensen"].notna()
            annual.loc[is_sarima, "pred_annual"] = annual.loc[is_sarima, "pred_annual_jensen"]
            annual = annual.drop(columns=["pred_annual_jensen"])
    annual["err_pct_model"] = (
        100.0 * (annual["pred_annual"] - annual["real_annual"]).abs()
        / annual["real_annual"].abs()
    )
    return annual


def _prefeitura_errors(cfg: PipelineConfig) -> pd.DataFrame:
    """Erro percentual da previsao da propria prefeitura, por (mun, tributo, ano)."""
    from forecasting.io import load_prefeitura_forecast
    pf = load_prefeitura_forecast(cfg)
    code_to_key = {m.cod_ibge: k for k, m in cfg.municipalities.items()}
    pf = pf[pf["cod_ibge"].isin(code_to_key)].copy()
    pf["municipio"] = pf["cod_ibge"].map(code_to_key)
    return pf[["municipio", "tributo", "year", "erro_pct_prefeitura"]].rename(
        columns={"year": "target_year"})


def municipality_benchmark_table(cfg: PipelineConfig) -> Path:
    """Gera tab_municipality_benchmark.tex.

    Para cada (municipio, tributo, modelo), conta em quantos dos anos-teste a
    previsao anual do modelo (soma das doze previsoes mensais) teve erro
    percentual menor que a previsao da propria prefeitura registrada na LOA, e
    reporta o erro percentual medio do modelo. Multiplos anos-teste (origem
    movel), em contraste com o ponto unico de Oliveira (2024).
    """
    from forecasting.evaluation import load_cv
    from forecasting.io import table_path

    annual = aggregate_monthly_to_annual(load_cv(cfg), cfg)
    pref = _prefeitura_errors(cfg)
    merged = annual.merge(pref, on=["municipio", "tributo", "target_year"], how="inner")
    merged["beat"] = merged["err_pct_model"] < merged["erro_pct_prefeitura"]

    # Conferencia com a prosa da Secao 5.6 (mesmo padrao [conferencia] da
    # generalizacao): os agregados 24/30 (80%) e 22/30 (73%) citados no texto
    # saem daqui, nao de soma manual sobre a tabela.
    ens = merged[merged["modelo"] == "Ensemble"]
    expost = 0
    for (_mk, _tr), b in merged.groupby(["municipio", "tributo"]):
        best = b.groupby("modelo")["err_pct_model"].mean().idxmin()
        expost += int(b[b["modelo"] == best]["beat"].sum())
    n_conf = ens["beat"].count()
    print(f"[conferencia] nucleo: Ensemble fixo={int(ens['beat'].sum())}/{n_conf} "
          f"({100 * ens['beat'].mean():.0f}%)  melhor ex-post={expost}/{n_conf} "
          f"({100 * expost / n_conf:.0f}%)")

    rows: list[str] = []
    keys = series_keys(cfg)
    for i, (mk, name, trib) in enumerate(keys):
        block = merged[(merged["municipio"] == mk) & (merged["tributo"] == trib)]
        n_years = block["target_year"].nunique()
        pref_err = block.groupby("target_year")["erro_pct_prefeitura"].first().mean()
        errs = {m: block[block["modelo"] == m]["err_pct_model"].mean()
                for m in MODEL_ORDER if not block[block["modelo"] == m].empty}
        best_err_disp = format_dec(min(errs.values()), 1) if errs else ""
        first = True
        for m in MODEL_ORDER:
            mb = block[block["modelo"] == m]
            if mb.empty:
                continue
            beat = int(mb["beat"].sum())
            err = mb["err_pct_model"].mean()
            err_cell = format_dec(err, 1)
            # Negrito no menor erro medio da serie (inclui empates ao display).
            if err_cell == best_err_disp:
                err_cell = f"\\textbf{{{err_cell}}}"
            head = f"{name} {trib}" if first else ""
            pref_cell = format_dec(pref_err, 1) if first else ""
            first = False
            rows.append(
                f"{head} & {MODEL_TEX[m]} & {beat}/{n_years} & "
                f"{err_cell} & {pref_cell} \\\\")
        if i < len(keys) - 1:
            rows.append(r"\addlinespace")
    from forecasting.config import styled_table
    tex = styled_table(
        gerado_por="benchmarks.municipality_benchmark_table",
        caption="Compara\\c{c}\\~ao anual com a previs\\~ao da prefeitura",
        label="tab:municipality-benchmark",
        colspec="l l r r r",  # numericos a direita: virgulas alinham na vertical
        header=["S\\'erie", "Modelo", "Venceu (anos)", "Erro m\\'edio (\\%)",
                "Erro prefeitura (\\%)"],
        rows=rows,
        fonte=("Elabora\\c{c}\\~ao pr\\'opria; prefeitura: RREO-Anexo 03, coluna "
               "Previs\\~ao Atualizada do primeiro bimestre (P1, a mais pr\\'oxima "
               "da Previs\\~ao Inicial da LOA)."),
        footnote=("Erro da previs\\~ao anual (soma das doze mensais) frente ao "
                  "realizado; negrito: menor erro m\\'edio da s\\'erie. ``Venceu'': "
                  "anos, dos cinco de teste (2021--2025), com erro abaixo do da "
                  "prefeitura."),
        stripe=True,
        # scriptsize/1.0: a tabela tem 36 linhas de dados e precisa caber,
        # JUNTO com a nota e a fonte, em UMA pagina do Apendice B (tabularx
        # nao quebra entre paginas; o aparato nao pode se separar dela).
        size="scriptsize",
        arraystretch="1.0",
        floating=False,  # vive no Apendice B, cuja mecanica e nao-flutuante
    )
    out = table_path(cfg, "tab_municipality_benchmark")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def best_model_per_series(cfg: PipelineConfig) -> dict[tuple[str, str], str]:
    """Modelo de menor erro percentual ANUAL medio por (municipio_key, tributo).

    Para o confronto anual com a prefeitura, o criterio relevante e a acuracia
    da previsao do TOTAL do exercicio (soma das doze previsoes mensais), nao a
    acuracia mes a mes (MASE). Seleciona, para cada serie, o modelo cujo erro
    percentual anual medio nos anos-teste e o menor -- a escolha apples-to-apples
    para o benchmark anual.
    """
    from forecasting.evaluation import load_cv
    annual = aggregate_monthly_to_annual(load_cv(cfg), cfg)
    best: dict[tuple[str, str], str] = {}
    for (mk, trib), block in annual.groupby(["municipio", "tributo"]):
        best[(mk, trib)] = block.groupby("modelo")["err_pct_model"].mean().idxmin()
    return best


# ---------- Confronto com Oliveira (2024) -------------------------------


def _our_2022_errors(cfg: PipelineConfig) -> dict[tuple[str, str], float]:
    """Erro anual em 2022 (h=12 agregado) do melhor modelo, por (mun, tributo)."""
    from forecasting.evaluation import load_cv
    annual = aggregate_monthly_to_annual(load_cv(cfg), cfg)
    best = best_model_per_series(cfg)
    out: dict[tuple[str, str], float] = {}
    for (mk, trib), bm in best.items():
        row = annual[(annual["municipio"] == mk) & (annual["tributo"] == trib) &
                     (annual["modelo"] == bm) & (annual["target_year"] == 2022)]
        if not row.empty:
            out[(mk, trib)] = float(row["err_pct_model"].iloc[0])
    return out


def oliveira_confronto_table(cfg: PipelineConfig) -> Path:
    """Gera tab_confronto_oliveira.tex: confronto direto, em 2022, do erro
    percentual de cada metodo de Oliveira (2024) --- previsao da prefeitura,
    Box-Jenkins, Holt-Winters e rede neural NNAR --- com o do modelo
    selecionado deste estudo (h=12 agregado), por serie. Comparacao
    descritiva: protocolos (ponto unico vs origem movel; anual vs mensal
    agregado) e deflatores (IGP-M vs IPCA) distintos.
    """
    from forecasting.io import table_path

    ours = _our_2022_errors(cfg)
    best = best_model_per_series(cfg)
    keys = series_keys(cfg)

    def _cell(v: float, lo: float, suffix: str = "") -> str:
        s = format_dec(v, 2) + suffix
        return f"\\textbf{{{s}}}" if abs(v - lo) < 1e-9 else s

    rows: list[str] = []
    for i, (mk, name, trib) in enumerate(keys):
        pref = OLIVEIRA_2022_ERRORS_PCT[(mk, trib)]
        met = OLIVEIRA_2022_METHOD_ERRORS_PCT[(mk, trib)]
        bj, hw, nnar = met["bj"], met["hw"], met["nnar"]
        our = ours.get((mk, trib), float("nan"))
        bm = best.get((mk, trib), "--")
        lo = min(pref, bj, hw, nnar, our)
        rows.append(
            f"{name} {trib} & {_cell(pref, lo)} & "
            f"{_cell(bj, lo)} & {_cell(hw, lo)} & {_cell(nnar, lo)} & "
            f"{_cell(our, lo, '~(' + MODEL_TEX[bm] + ')')} \\\\")
        if trib == "ISSQN" and i < len(keys) - 1:
            rows.append(r"\addlinespace")
    from forecasting.config import styled_table
    tex = styled_table(
        gerado_por="benchmarks.oliveira_confronto_table",
        caption="Compara\\c{c}\\~ao direta com Oliveira (2024)",
        label="tab:confronto-oliveira",
        colspec="L r r r r L",
        header=["S\\'erie", "Prefeitura", "Box-Jenkins", "Holt-Winters", "NNAR",
                "Este estudo"],
        rows=rows,
        fonte=("Erros dos m\\'etodos de \\citeonline{oliveira2024} transcritos das "
               "Tabelas 01--10 daquele trabalho; a coluna ``Este estudo'' foi "
               "elaborada pelo autor."),
        footnote=("Erros percentuais absolutos do total de 2022. A coluna ``Este "
                  "estudo'' usa, em cada s\\'erie, o modelo de menor erro anual no "
                  "per\\'iodo (sele\\c{c}\\~ao \\emph{ex-post}, indicada entre par\\^enteses)."),
        stripe=True,
        size="footnotesize",
    )
    out = table_path(cfg, "tab_confronto_oliveira")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def prefeitura_ensemble_figure(cfg: PipelineConfig) -> Path:
    """Gera fig_confronto_prefeitura.pdf (dumbbell): erro anual medio da
    previsao da PREFEITURA versus o do ENSEMBLE, por serie, nos anos-teste.

    Mesmos dados da tab_municipality_benchmark (cache -> agregado anual ->
    media por serie): a figura ancora a leitura da tabela. Series ordenadas
    pelo erro da prefeitura, para que o padrao 'o ganho se concentra onde a
    projecao oficial e fragil' salte aos olhos. Cache-only."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from forecasting.config import MUNITAX_BLUE
    from forecasting.evaluation import load_cv
    from forecasting.plotting import save_figure, setup_matplotlib_thesis, style_axis

    setup_matplotlib_thesis()
    annual = aggregate_monthly_to_annual(load_cv(cfg), cfg)
    pref = _prefeitura_errors(cfg)
    merged = annual.merge(pref, on=["municipio", "tributo", "target_year"],
                          how="inner")

    rows = []
    for (mk, name, trib) in series_keys(cfg):
        block = merged[(merged["municipio"] == mk) & (merged["tributo"] == trib)]
        pref_err = block.groupby("target_year")["erro_pct_prefeitura"].first().mean()
        ens_err = block[block["modelo"] == "Ensemble"]["err_pct_model"].mean()
        rows.append((f"{name} · {trib}", float(pref_err), float(ens_err)))
    rows.sort(key=lambda r: r[1])  # menor erro da prefeitura embaixo

    pref_c = "#9AA0A6"
    ens_c = MUNITAX_BLUE
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ys = np.arange(len(rows))
    for y, (_lab, p, e) in zip(ys, rows):
        ax.plot([min(p, e), max(p, e)], [y, y], color="#DFE3E8", lw=2.6,
                solid_capstyle="round", zorder=1)
        ax.scatter([p], [y], s=42, color=pref_c, zorder=3)
        ax.scatter([e], [y], s=42, color=ens_c, zorder=4)
        gap = abs(p - e)
        for v, c in ((p, "#5F6368"), (e, ens_c)):
            # Rotulo de valor sempre INLINE no lado externo de cada ponto (rotulo
            # do menor a esquerda, do maior a direita); so empilha na vertical se
            # os pontos quase coincidem (raro), quando nao ha lado externo claro.
            if gap < 1.0:
                ax.annotate(f"{v:.1f}".replace(".", ","), (v, y),
                            xytext=(0, 7 if c == ens_c else -8),
                            textcoords="offset points", ha="center",
                            va="bottom" if c == ens_c else "top",
                            fontsize=7.0, color=c, fontweight="semibold")
            else:
                dx = 0.55 * (1 if v == max(p, e) else -1)
                ax.annotate(f"{v:.1f}".replace(".", ","), (v + dx, y),
                            ha="left" if dx > 0 else "right", va="center",
                            fontsize=7.0, color=c, fontweight="semibold")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("Erro anual médio (%), 2021 a 2025")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.14)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(False)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0.0)
    style_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=6, color=pref_c,
               label="Previsão da prefeitura (LOA)"),
        Line2D([0], [0], marker="o", lw=0, markersize=6, color=ens_c,
               label="Ensemble"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               fontsize=7.7, frameon=False)
    out = save_figure(fig, "fig_confronto_prefeitura", cfg.figures_dir_abs)
    plt.close(fig)
    return out


# ---------- Orquestracao --------------------------------------------------


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Executa benchmark da prefeitura e confronto com Oliveira (Secao 5.5)."""
    return [
        municipality_benchmark_table(cfg),
        prefeitura_ensemble_figure(cfg),
        oliveira_confronto_table(cfg),
    ]
