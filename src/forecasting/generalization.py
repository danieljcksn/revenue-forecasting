"""Analise de generalizacao: estende o estudo aos municipios baianos populosos.

O nucleo do TCC examina em profundidade tres municipios de perfis economicos
contrastantes (Salvador, Camacari, Ilheus). Esta camada acrescenta BREADTH:
aplica o mesmo pipeline a todos os municipios baianos com mais de cem mil
habitantes --- o mesmo recorte de Oliveira (2024) --- para verificar se os
padroes observados nos tres casos se sustentam num conjunto mais amplo. O
objetivo nao e detalhar cada serie, mas medir, no agregado: (i) com que
frequencia cada modelo vence; (ii) se a previsibilidade IPTU > ISSQN persiste;
(iii) a taxa de superacao da previsao da propria prefeitura.

Tratamento de qualidade: cada serie passa por um detector de anos anomalos
(total zero ou abaixo de 55% da media dos anos adjacentes). Anos isolados sao
imputados por sazonal naive a partir dos exercicios vizinhos; series com mais
de dois anos comprometidos sao excluidas e o fato e registrado, para nao
contaminar o agregado.
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
    MODEL_TEX,
    PipelineConfig,
    format_dec,
)

warnings.filterwarnings("ignore")

# Municipios baianos com mais de 100 mil habitantes (IBGE 2022) -- recorte de
# Oliveira (2024). (cod_ibge, nome).
POPULOUS_BA: list[tuple[int, str]] = [
    (2927408, "Salvador"), (2910800, "Feira de Santana"),
    (2933307, "Vitória da Conquista"), (2905701, "Camaçari"),
    (2914802, "Itabuna"), (2918407, "Juazeiro"), (2919207, "Lauro de Freitas"),
    (2913606, "Ilhéus"), (2918001, "Jequié"), (2931350, "Teixeira de Freitas"),
    (2900702, "Alagoinhas"), (2903201, "Barreiras"), (2925303, "Porto Seguro"),
    (2930709, "Simões Filho"), (2924009, "Paulo Afonso"), (2910727, "Eunápolis"),
    (2928604, "Santo Antônio de Jesus"), (2932507, "Valença"), (2906501, "Candeias"),
]

# --- Limiares do controle de qualidade das series estendidas (decisoes
#     metodologicas; mesmos valores de antes, agora nomeados e justificados). ---
_MIN_COBERTURA_MESES = 120  # >= 120 meses (10 anos) de cobertura p/ a serie entrar
_ANO_COMPLETO_MESES = 10    # so anos com >= 10 meses contam no teste de queda de nivel
_QUEDA_NIVEL_FRAC = 0.55    # ano "cai de nivel" se ficar < 55% da media dos vizinhos
_MAX_ANOS_ANOMALOS = 2      # > 2 anos anomalos -> serie inutilizavel (exclui)
_INTERP_MAX_GAP = 2         # interpola lacunas de ate 2 meses; gaps maiores excluem


def _detect_anomalous_years(s: pd.Series) -> list[int]:
    """Anos totalmente ausentes/nulos, ou com queda de nivel abrupta (abaixo de
    55% da media dos anos adjacentes). A compara\\c{c}ao de nivel so se aplica a
    anos com cobertura quase completa (>= 10 meses): assim um ano apenas
    esparso nao e confundido com uma queda de nivel e indevidamente imputado."""
    ann = s.groupby(s.index.year).sum()
    cnt = s.groupby(s.index.year).count()
    bad = []
    for y in ann.index:
        adj = [ann.get(y - 1), ann.get(y + 1)]
        adj = [a for a in adj if a is not None and not np.isnan(a) and a > 0]
        all_missing = cnt[y] == 0 or ann[y] <= 0
        level_drop = (cnt[y] >= _ANO_COMPLETO_MESES and bool(adj)
                      and ann[y] < _QUEDA_NIVEL_FRAC * np.mean(adj))
        if all_missing or level_drop:
            bad.append(int(y))
    return bad


def prepare_extended_series(cfg: PipelineConfig, municipios=None):
    """Series mensais deflacionadas dos municipios populosos, com tratamento
    automatico de anomalias. Retorna (series, log) onde ``series`` mapeia
    (cod_ibge, nome, tributo) -> Series e ``log`` documenta o tratamento.

    ``municipios`` (lista de pares ``(cod_ibge, nome)``) substitui o recorte
    padrao POPULOUS_BA; e usado pela analise estadual em ``analysis/bahia``."""
    from forecasting.eda import deflate_by_ipca, impute_anomalous_year
    from forecasting.io import load_monthly_series, tributo_column

    raw = load_monthly_series(cfg)
    defl = deflate_by_ipca(raw, base_month=cfg.ipca_base_month)
    start = pd.Timestamp(cfg.sample_window.start + "-01")
    end = pd.Timestamp(cfg.sample_window.end + "-01")

    series: dict[tuple, pd.Series] = {}
    log = {"imputed": [], "interpolated": [], "excluded": [], "absent": []}
    present = set(defl["cod_ibge"].unique())

    for cod, nome in (municipios if municipios is not None else POPULOUS_BA):
        if cod not in present:
            log["absent"].append(nome)
            continue
        sub = defl[defl["cod_ibge"] == cod].sort_values("date")
        for tributo in cfg.tributos:
            col = tributo_column(tributo)
            s = pd.Series(
                pd.to_numeric(sub[col], errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(sub["date"]).to_period("M").to_timestamp(),
            )
            s = s.loc[(s.index >= start) & (s.index <= end)].asfreq("MS")
            if s.notna().sum() < _MIN_COBERTURA_MESES:
                log["excluded"].append((nome, tributo, "cobertura < 120 meses"))
                continue
            # Valores mensais nao-positivos (estornos/retificacoes contabeis)
            # sao incompativeis com a transformacao log do SARIMA: log(neg)=NaN,
            # que o statsmodels absorve silenciosamente como dado faltante,
            # contaminando as metricas de forma dependente da posicao. Excluem-se
            # essas series de forma explicita e simetrica para os quatro modelos.
            if (s <= 0).any():
                log["excluded"].append((nome, tributo, "valor mensal nao-positivo"))
                continue
            bad = _detect_anomalous_years(s)
            if len(bad) > _MAX_ANOS_ANOMALOS:
                log["excluded"].append((nome, tributo, f"{len(bad)} anos anomalos"))
                continue
            for yr in bad:
                s = impute_anomalous_year(s, yr)
                log["imputed"].append((nome, tributo, yr))
            # lacunas isoladas (ate 2 meses consecutivos): interpolacao linear;
            # gaps maiores tornam a serie inutilizavel -> exclui.
            n_nan = int(s.isna().sum())
            if n_nan:
                s = s.interpolate(method="linear", limit=_INTERP_MAX_GAP, limit_area="inside")
                if s.isna().any():
                    log["excluded"].append((nome, tributo, f"{n_nan} meses ausentes"))
                    continue
                log["interpolated"].append((nome, tributo, n_nan))
            series[(cod, nome, tributo)] = s
    return series, log


def _log(msg: str) -> None:
    """Mensagem de progresso para o console (stdout, com flush imediato)."""
    print(msg, flush=True)


def run_generalization(cfg: PipelineConfig) -> Path:
    """Roda a validacao por origem movel (4 modelos) em todas as series
    estendidas e cacheia o resultado em ``cfg.forecasts_dir / cv_extended.csv``."""
    from forecasting import models as M

    series, log = prepare_extended_series(cfg)
    log.setdefault("fit_failed", [])
    frames = []
    for (cod, nome, tributo), s in series.items():
        # Isola cada serie: uma falha de ajuste (modelo que diverge numa janela
        # expandida e produz NaN/inf) e registrada e a serie e descartada
        # inteira, para que toda serie remanescente compare os quatro modelos
        # sob exatamente as mesmas dobras. Nao se contamina o agregado.
        try:
            fitters = M.make_fitters(s)
            series_frames = []
            for mname, fn in fitters.items():
                cv = M.rolling_origin_cv(
                    s, fn, initial_window=M.INITIAL_WINDOW,
                    max_horizon=M.MAX_HORIZON, step=M.ROLLING_STEP)
                if not np.isfinite(cv["y_pred"].to_numpy()).all():
                    raise ValueError(f"{mname} produziu previsao nao-finita")
                cv.insert(0, "modelo", mname)
                cv.insert(0, "tributo", tributo)
                cv.insert(0, "municipio_nome", nome)
                cv.insert(0, "cod_ibge", cod)
                series_frames.append(cv)
        except Exception as exc:  # noqa: BLE001 -- robustez do lote
            log["fit_failed"].append((nome, tributo, f"{type(exc).__name__}: {exc}"))
            _log(f"[skip] {nome:24s} {tributo} -> {type(exc).__name__}: {exc}")
            continue
        frames.extend(series_frames)
        _log(f"[ok] {nome:24s} {tributo}")
    cv_all = pd.concat(frames, ignore_index=True)
    out = cfg.forecasts_dir / "cv_extended.csv"
    cv_all.to_csv(out, index=False, encoding="utf-8")
    # log de tratamento
    n_ok = cv_all[["cod_ibge", "tributo"]].drop_duplicates().shape[0]
    (cfg.forecasts_dir / "extended_log.txt").write_text(
        f"imputed: {log['imputed']}\ninterpolated: {log['interpolated']}\n"
        f"excluded: {log['excluded']}\nabsent: {log['absent']}\n"
        f"fit_failed: {log['fit_failed']}\nseries_avaliadas: {n_ok}\n",
        encoding="utf-8")
    _log(f"\nseries preparadas: {len(series)} | avaliadas: {n_ok} | "
         f"imputadas: {len(log['imputed'])} | "
         f"interpoladas: {len(log['interpolated'])} | "
         f"excluidas: {len(log['excluded'])} | falha-ajuste: "
         f"{len(log['fit_failed'])} | ausentes: {len(log['absent'])}")
    return out


def _load_extended_cv(cfg: PipelineConfig) -> pd.DataFrame:
    path = cfg.forecasts_dir / "cv_extended.csv"
    cv = pd.read_csv(path, parse_dates=["origin", "train_end", "target_date"])
    cv["scaled_err"] = (cv["y_true"] - cv["y_pred"]).abs() / cv["insample_scale"]
    return cv


def generalization_municipality_table(cfg: PipelineConfig) -> Path:
    """tab_generalizacao_municipios.tex: o melhor modelo e seu MASE mediano
    (h=12) de CADA municipio baiano populoso, por tributo. Torna concreta a
    amplitude da analise --- os dezoito municipios, nominalmente."""
    from forecasting.io import table_path

    cv = _load_extended_cv(cfg)
    h12 = cv[cv["step"] == 12]
    med = (h12.groupby(["municipio_nome", "tributo", "modelo"])["scaled_err"]
           .median().reset_index())
    def cell(nome: str, trib: str) -> str:
        g = med[(med["municipio_nome"] == nome) & (med["tributo"] == trib)]
        if g.empty:
            return "--"
        w = g.loc[g["scaled_err"].idxmin()]
        val = format_dec(w['scaled_err'], 2)
        return f"{MODEL_TEX[w['modelo']]} ({val})"

    from forecasting.config import styled_table

    muns = sorted(med["municipio_nome"].unique())
    rows = [f"{nome} & {cell(nome, 'IPTU')} & {cell(nome, 'ISSQN')} \\\\"
            for nome in muns]
    tex = styled_table(
        gerado_por="generalization.generalization_municipality_table",
        caption="Melhor modelo por munic\\'ipio e tributo",
        label="tab:generalizacao-municipios",
        colspec="L L L",
        header=["Munic\\'ipio", "IPTU: melhor (MASE)", "ISSQN: melhor (MASE)"],
        rows=rows,
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        footnote=("Entre par\\^enteses, o MASE mediano em $h=12$. ``--'' indica "
                  "s\\'erie exclu\\'ida no controle de qualidade."),
        stripe=True,
        size="footnotesize",
        floating=False,
    )
    out = table_path(cfg, "tab_generalizacao_municipios")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def generalization_table(cfg: PipelineConfig) -> Path:
    """tab_generalizacao.tex: no conjunto dos municipios populosos, distribuicao
    de vitorias por modelo (MASE mediano, h=12) e MASE tipico, por tributo."""
    from forecasting.io import table_path

    cv = _load_extended_cv(cfg)
    h12 = cv[cv["step"] == 12]
    # MASE mediano por (municipio, tributo, modelo)
    med = h12.groupby(["municipio_nome", "tributo", "modelo"])["scaled_err"].median().reset_index()
    rows = []
    for tributo in ["IPTU", "ISSQN"]:
        sub = med[med["tributo"] == tributo]
        n_series = sub["municipio_nome"].nunique()
        # vencedor por serie
        wins = {m: 0 for m in MODEL_ORDER}
        best_vals = []
        for _, g in sub.groupby("municipio_nome"):
            w = g.loc[g["scaled_err"].idxmin()]
            wins[w["modelo"]] += 1
            best_vals.append(w["scaled_err"])
        # MASE mediano do melhor modelo, e mediana geral por modelo
        max_wins = max(wins.values())  # negrito no modelo com mais vitorias (inclui empates)
        win_cells = [
            f"\\textbf{{{wins[m]}}}" if wins[m] == max_wins and wins[m] > 0
            else str(wins[m])
            for m in MODEL_ORDER
        ]
        med_best = np.median(best_vals)
        rows.append(
            " & ".join([tributo, str(n_series), *win_cells, format_dec(med_best, 2)])
            + " \\\\")
    from forecasting.config import styled_table

    tex = styled_table(
        gerado_por="generalization.generalization_table",
        caption="Generaliza\\c{c}\\~ao aos munic\\'ipios populosos",
        label="tab:generalizacao",
        # "Tributo", "$n$" e "MASE" em largura natural; as colunas de modelo
        # dividem o restante (senao o cabecalho "Ensemble" negrito estoura).
        colspec="l c C C C C C C c",
        header=["Tributo", "$n$", *[MODEL_TEX[m] for m in MODEL_ORDER], "MASE"],
        rows=rows,
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        footnote=("$n$ indica o n\\'umero de s\\'eries retidas no controle de qualidade. "
                  "As colunas dos modelos contam as vit\\'orias por menor MASE mediano "
                  "em $h=12$; em negrito, o maior total de cada linha (inclusive "
                  "empates). A coluna MASE mostra a mediana dos vencedores."),
        stripe=True,
        size="footnotesize",
    )
    out = table_path(cfg, "tab_generalizacao")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def generalization_figure(cfg: PipelineConfig) -> Path:
    """fig_generalizacao.pdf: MASE mediano (h=12) por modelo no conjunto
    estendido, separado por tributo, com linha de referencia em MASE=1."""
    import matplotlib.pyplot as plt

    from forecasting.plotting import (
        BAR_LABEL_SIZE,
        BASELINE_GREY,
        br_axis,
        pixel_scale,
        rounded_bar,
        save_figure,
        setup_matplotlib_thesis,
        style_axis,
    )

    setup_matplotlib_thesis()
    cv = _load_extended_cv(cfg)
    h12 = cv[cv["step"] == 12]
    med = (h12.groupby(["tributo", "municipio_nome", "modelo"])["scaled_err"]
           .median().reset_index()
           .groupby(["tributo", "modelo"])["scaled_err"].median().reset_index())
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.35), sharex=True, sharey=True)
    ymax = max(1.15, float(med["scaled_err"].max()) * 1.18)
    x = np.arange(len(MODEL_ORDER))
    width = 0.62
    for ax, tributo in zip(axes, ["IPTU", "ISSQN"]):
        vals = [
            float(med[(med["tributo"] == tributo) & (med["modelo"] == m)]["scaled_err"].iloc[0])
            for m in MODEL_ORDER
        ]
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.65, len(MODEL_ORDER) - 0.35)
        ax.axhline(1.0, color=BASELINE_GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
        scale = pixel_scale(ax)
        for xpos, val, model in zip(x, vals, MODEL_ORDER):
            rounded_bar(ax, xpos - width / 2, 0, width, val,
                        MODEL_COLORS[model], scale=scale)
            ax.text(
                xpos, val + ymax * 0.025, f"{val:.2f}".replace(".", ","),
                ha="center", va="bottom", fontsize=BAR_LABEL_SIZE,
                fontweight="semibold", color="#202124",
            )
        ax.set_title(tributo)
        ax.set_ylabel("MASE mediano")
        # Passo regular de 0,5: o locator automatico, arredondado a 1 casa,
        # produzia a sequencia irregular 0,0 / 0,2 / 0,5 / 0,8 / 1,0.
        br_axis(ax, "y", decimals=1, step=0.5)
        style_axis(ax)
    axes[-1].set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER])
    axes[-1].tick_params(axis="x", length=0.0)
    out = save_figure(fig, "fig_generalizacao", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def generalization_prefeitura_figure(cfg: PipelineConfig) -> Path:
    """Gera fig_confronto_ampliado.pdf (dumbbell, 31 series): erro anual medio
    do ENSEMBLE contra o da PREFEITURA no conjunto ampliado, mesmo desenho da
    fig_confronto_prefeitura para leitura continua entre as duas.

    Reproduz a MESMA regra do texto (origens que terminam em dezembro, soma dos
    doze passos, anos com previsao municipal disponivel) e imprime os win-rates
    para conferencia com a prosa (canonicos P1: 155 confrontos, ex-post 70%,
    Ensemble fixo 63%; Ensemble com erro medio menor em 24 das 31 series).
    Series ordenadas pelo erro da prefeitura; conector azulado onde o Ensemble
    vence. Cache-only."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from forecasting.config import MUNITAX_BLUE
    from forecasting.io import load_prefeitura_forecast
    from forecasting.plotting import save_figure, setup_matplotlib_thesis, style_axis

    setup_matplotlib_thesis()
    cv = _load_extended_cv(cfg)
    cv = cv.copy()
    cv["origin"] = pd.to_datetime(cv["origin"])
    dec = cv[(cv["origin"].dt.month == 12) & (cv["step"].between(1, 12))].copy()
    dec["target_year"] = dec["origin"].dt.year + 1
    g = dec.groupby(["cod_ibge", "tributo", "modelo", "target_year"]).agg(
        pred=("y_pred", "sum"), real=("y_true", "sum"), n=("step", "count")).reset_index()
    g = g[g["n"] == 12]
    g["err"] = 100 * (g["pred"] - g["real"]).abs() / g["real"].abs()

    pf = load_prefeitura_forecast(cfg).rename(columns={"year": "target_year"})
    m = g.merge(pf[["cod_ibge", "tributo", "target_year", "erro_pct_prefeitura"]],
                on=["cod_ibge", "tributo", "target_year"], how="inner")

    # Conferencia com a prosa do par. 5.7 (nao altera a figura).
    m["beat"] = m["err"] < m["erro_pct_prefeitura"]
    n_conf = m[["cod_ibge", "tributo", "target_year"]].drop_duplicates().shape[0]
    expost = 0
    for (_c, _t), b in m.groupby(["cod_ibge", "tributo"]):
        best = b.groupby("modelo")["err"].mean().idxmin()
        expost += int(b[b["modelo"] == best]["beat"].sum())
    ens_fix = int(m[m["modelo"] == "Ensemble"]["beat"].sum())
    print(f"[conferencia] confrontos={n_conf}  ex-post={expost} "
          f"({100*expost/n_conf:.0f}%)  Ensemble fixo={ens_fix} "
          f"({100*ens_fix/n_conf:.0f}%)")

    ens = m[m["modelo"] == "Ensemble"]
    pts = ens.groupby(["cod_ibge", "tributo"]).agg(
        modelo_err=("err", "mean"),
        pref_err=("erro_pct_prefeitura", "mean")).reset_index()
    wins = int((pts["modelo_err"] < pts["pref_err"]).sum())
    print(f"[conferencia] series no grafico={len(pts)}; Ensemble com erro medio "
          f"menor que a prefeitura em {wins} delas")

    nomes = cv[["cod_ibge", "municipio_nome"]].drop_duplicates().set_index("cod_ibge")
    pts["nome"] = pts["cod_ibge"].map(nomes["municipio_nome"])
    pts["rotulo"] = pts["nome"] + " · " + pts["tributo"]
    pts = pts.sort_values("pref_err").reset_index(drop=True)

    pref_c = "#9AA0A6"
    ens_c = MUNITAX_BLUE
    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    ys = np.arange(len(pts))
    for y, row in pts.iterrows():
        p, e = float(row["pref_err"]), float(row["modelo_err"])
        seg_c = "#CBE2FB" if e < p else "#E8EAED"
        ax.plot([min(p, e), max(p, e)], [y, y], color=seg_c, lw=2.4,
                solid_capstyle="round", zorder=1)
        ax.scatter([p], [y], s=26, color=pref_c, zorder=3)
        ax.scatter([e], [y], s=26, color=ens_c, zorder=4)
    ax.set_yticks(ys, pts["rotulo"])
    ax.tick_params(axis="y", labelsize=7.0, length=0.0)
    ax.set_xlabel("Erro anual médio (%), 2021 a 2025")
    ax.set_xlim(0, float(max(pts["pref_err"].max(), pts["modelo_err"].max())) * 1.06)
    ax.set_ylim(-0.7, len(pts) - 0.3)
    ax.grid(False)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.6)
    ax.set_axisbelow(True)
    style_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=6, color=pref_c,
               label="Previsão da prefeitura (LOA)"),
        Line2D([0], [0], marker="o", lw=0, markersize=6, color=ens_c,
               label="Ensemble"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               fontsize=7.7, frameon=False)
    out = save_figure(fig, "fig_confronto_ampliado", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Gera as duas tabelas e as duas figuras vivas do conjunto ampliado.

    ATENCAO: se cv_extended.csv sumir, o re-fit automatico usaria o nucleo de
    QUATRO modelos de make_fitters, nao o portfolio canonico de seis gerado em
    scripts de geracao do conjunto ampliado -- nunca apague o cache."""
    if not (cfg.forecasts_dir / "cv_extended.csv").exists():
        run_generalization(cfg)
    return [generalization_table(cfg), generalization_municipality_table(cfg),
            generalization_figure(cfg), generalization_prefeitura_figure(cfg)]
