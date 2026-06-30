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
    origem/passo, escala log; produzido por _precisao_run/f4_jensen.py). Retorna
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
        caption=(
            "Confronto anual com a previs\\~ao da pr\\'opria prefeitura "
            "(LOA): anos em que o modelo superou a prefeitura e erro percentual "
            "m\\'edio, por s\\'erie."),
        label="tab:municipality-benchmark",
        colspec="l l C C C",
        header=["S\\'erie", "Modelo", "Venceu (anos)", "Erro m\\'edio (\\%)",
                "Erro prefeitura (\\%)"],
        rows=rows,
        fonte=("Elabora\\c{c}\\~ao pr\\'opria; previs\\~ao da prefeitura do "
               "RREO-Anexo 03 (Previs\\~ao Atualizada)."),
        footnote=("Erro percentual da previs\\~ao anual "
                  "(soma das doze previs\\~oes mensais em origem que termina em dezembro) "
                  "frente ao realizado; em negrito, o menor erro m\\'edio de cada s\\'erie. "
                  "``Venceu'': anos com erro do modelo abaixo do "
                  "erro da prefeitura, sobre os anos-teste 2021--2025."),
        stripe=True,
        size="footnotesize",
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


def municipality_vs_models_plot(cfg: PipelineConfig) -> Path:
    """Gera fig_municipality_vs_models.pdf.

    Para cada serie (painel 3x2), o erro percentual anual da previsao da
    prefeitura e o do melhor modelo, ano a ano (2021--2025).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from forecasting.evaluation import load_cv
    from forecasting.plotting import save_figure, setup_matplotlib_thesis

    setup_matplotlib_thesis()
    annual = aggregate_monthly_to_annual(load_cv(cfg), cfg)
    pref = _prefeitura_errors(cfg)
    best = best_model_per_series(cfg)

    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.4), sharex=True)
    keys = [(mk, trib) for mk, _name, trib in series_keys(cfg)]
    for ax, (mk, trib) in zip(axes.flat, keys):
        bm = best[(mk, trib)]
        ma = annual[(annual["municipio"] == mk) & (annual["tributo"] == trib) &
                    (annual["modelo"] == bm)].sort_values("target_year")
        pe = pref[(pref["municipio"] == mk) & (pref["tributo"] == trib)]
        merged = ma.merge(pe, on=["municipio", "tributo", "target_year"])
        x = np.arange(len(merged))
        ax.bar(x - 0.2, merged["erro_pct_prefeitura"], 0.4, label="Prefeitura", color="#BBBBBB")
        ax.bar(x + 0.2, merged["err_pct_model"], 0.4,
               label="Melhor modelo", color="#0072B2")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["target_year"].astype(int))
        ax.set_title(f"{mun_label(cfg)[mk]} · {trib} ({bm})")
        if ax in axes[:, 0]:
            ax.set_ylabel("Erro anual (\\%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    out = save_figure(fig, "fig_municipality_vs_models", cfg.figures_dir_abs)
    plt.close(fig)
    return out


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
        caption=(
            "Confronto direto com \\citeonline{oliveira2024}: erro "
            "percentual absoluto da previs\\~ao para o exerc\\'icio de 2022, por "
            "m\\'etodo e s\\'erie. Em negrito, o menor erro de cada linha."),
        label="tab:confronto-oliveira",
        colspec="L r r r r L",
        header=["S\\'erie", "Prefeitura", "Box-Jenkins", "Holt-Winters", "NNAR",
                "Este estudo"],
        rows=rows,
        fonte=("Erros dos m\\'etodos de \\citeonline{oliveira2024} (previs\\~ao "
               "da pr\\'opria prefeitura, Box-Jenkins, Holt-Winters e rede neural NNAR) "
               "transcritos das Tabelas 01--10 daquele trabalho; ``Este estudo'' "
               "\\'e o erro, em 2022, do modelo que, em retrospecto, melhor se ajustou "
               "a cada s\\'erie no per\\'iodo (sele\\c{c}\\~ao \\emph{ex-post}, indicada "
               "entre par\\^enteses), elaborado pelo autor."),
        footnote=("\\textsuperscript{} Os valores de "
                  "\\citeonline{oliveira2024} prov\\^em de avalia\\c{c}\\~ao em ponto "
                  "\\'unico (2022), com modelos \\emph{anuais} e deflator IGP-M; os deste "
                  "estudo agregam previs\\~oes \\emph{mensais} em origem m\\'ovel sob "
                  "IPCA. O confronto \\'e, portanto, descritivo, n\\~ao uma competi\\c{c}"
                  "\\~ao ponto a ponto. Acrescente-se que a coluna ``Este estudo'' usa, em "
                  "cada s\\'erie, o modelo de menor erro anual no per\\'iodo (sele\\c{c}\\~ao "
                  "\\emph{ex-post}, que inclui 2022): leitura ilustrativa, n\\~ao desempenho "
                  "assegur\\'avel \\emph{a priori}."),
        stripe=True,
        size="small",
    )
    out = table_path(cfg, "tab_confronto_oliveira")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def confronto_oliveira_plot(cfg: PipelineConfig) -> Path:
    """Gera fig_confronto_oliveira.pdf: barras emparelhadas (erro da prefeitura
    vs erro do melhor modelo deste estudo) por serie, em 2022."""
    import matplotlib.pyplot as plt
    import numpy as np

    from forecasting.plotting import save_figure, setup_matplotlib_thesis

    setup_matplotlib_thesis()
    ours = _our_2022_errors(cfg)
    keys = [(mk, trib) for mk, _name, trib in series_keys(cfg)]
    labels = [f"{mun_label(cfg)[mk][:3]}\n{trib}" for mk, trib in keys]
    pref_vals = [OLIVEIRA_2022_ERRORS_PCT[k] for k in keys]
    our_vals = [ours.get(k, float("nan")) for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(x - 0.2, pref_vals, 0.4, label="Prefeitura (Oliveira 2024)", color="#999999")
    ax.bar(x + 0.2, our_vals, 0.4, label="Melhor modelo (este estudo)", color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Erro percentual em 2022 (%)")
    ax.legend(fontsize=8)
    out = save_figure(fig, "fig_confronto_oliveira", cfg.figures_dir_abs)
    plt.close(fig)
    return out


# ---------- Validacoes cruzadas (asserts de sanidade) -------------------


def assert_annual_aggregation_matches_anexo01(
    cfg: PipelineConfig,
    tolerance: float = 0.02,
) -> list[str]:
    """Verifica se a soma dos doze meses da serie mensal (Anexo 03, nominal)
    reproduz o realizado anual do benchmark da prefeitura, dentro de tolerancia.

    Opera em valores NOMINAIS (a serie mensal sem deflacionar), pois o
    realizado anual da prefeitura tambem e nominal. Retorna a lista de
    divergencias acima da tolerancia (vazia se tudo confere)."""
    from forecasting.io import load_monthly_series, load_prefeitura_forecast, tributo_column

    raw = load_monthly_series(cfg)
    pf = load_prefeitura_forecast(cfg)
    warnings_list: list[str] = []
    for key, mun in cfg.municipalities.items():
        sub = raw[raw["cod_ibge"] == mun.cod_ibge]
        for tributo in cfg.tributos:
            col = tributo_column(tributo)
            monthly_annual = sub.groupby("year")[col].sum()
            pf_sub = pf[(pf["cod_ibge"] == mun.cod_ibge) & (pf["tributo"] == tributo)]
            for _, row in pf_sub.iterrows():
                yr = int(row["year"])
                if yr not in monthly_annual.index:
                    continue
                got = float(monthly_annual.loc[yr])
                ref = float(row["realizado_anual"])
                if ref and abs(got - ref) / abs(ref) > tolerance:
                    warnings_list.append(
                        f"{mun.name}/{tributo}/{yr}: soma mensal {got:.0f} vs "
                        f"realizado_anual {ref:.0f} (dif {100*abs(got-ref)/abs(ref):.1f}%)")
    return warnings_list


def assert_oliveira_proximity(
    our_2022_errors: dict[tuple[str, str], float],
    tolerance_pp: float = 2.0,
) -> list[str]:
    """Compara os erros da PREFEITURA para 2022 calculados aqui com os de
    Oliveira (2024). Diferenca > tolerance_pp e sinalizada. Retorna avisos."""
    warnings_list: list[str] = []
    for key, ref in OLIVEIRA_2022_ERRORS_PCT.items():
        got = our_2022_errors.get(key)
        if got is None:
            continue
        if abs(got - ref) > tolerance_pp:
            warnings_list.append(
                f"{key}: aqui {got:.2f}% vs Oliveira {ref:.2f}% "
                f"(dif {abs(got-ref):.2f} p.p.)")
    return warnings_list


# ---------- Orquestracao --------------------------------------------------


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Executa benchmark da prefeitura e confronto com Oliveira (Secao 5.5)."""
    return [
        municipality_benchmark_table(cfg),
        oliveira_confronto_table(cfg),
    ]
