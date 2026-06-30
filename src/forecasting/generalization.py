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


def prepare_extended_series(cfg: PipelineConfig):
    """Series mensais deflacionadas dos municipios populosos, com tratamento
    automatico de anomalias. Retorna (series, log) onde ``series`` mapeia
    (cod_ibge, nome, tributo) -> Series e ``log`` documenta o tratamento."""
    from forecasting.eda import deflate_by_ipca, impute_anomalous_year
    from forecasting.io import load_monthly_series, tributo_column

    raw = load_monthly_series(cfg)
    defl = deflate_by_ipca(raw, base_month=cfg.ipca_base_month)
    start = pd.Timestamp(cfg.sample_window.start + "-01")
    end = pd.Timestamp(cfg.sample_window.end + "-01")

    series: dict[tuple, pd.Series] = {}
    log = {"imputed": [], "interpolated": [], "excluded": [], "absent": []}
    present = set(defl["cod_ibge"].unique())

    for cod, nome in POPULOUS_BA:
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
            return "---"
        w = g.loc[g["scaled_err"].idxmin()]
        val = format_dec(w['scaled_err'], 2)
        return f"{MODEL_TEX[w['modelo']]} ({val})"

    from forecasting.config import styled_table

    muns = sorted(med["municipio_nome"].unique())
    rows = [f"{nome} & {cell(nome, 'IPTU')} & {cell(nome, 'ISSQN')} \\\\"
            for nome in muns]
    tex = styled_table(
        gerado_por="generalization.generalization_municipality_table",
        caption="Melhor modelo e seu MASE mediano em $h=12$ por munic\\'ipio "
        "baiano com mais de cem mil habitantes, por tributo (entre par\\^enteses, "
        "o MASE). ``---'' indica s\\'erie exclu\\'ida no controle de qualidade.",
        label="tab:generalizacao-municipios",
        colspec="L L L",
        header=["Munic\\'ipio", "IPTU: melhor (MASE)", "ISSQN: melhor (MASE)"],
        rows=rows,
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=True,
        size="small",
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
        win_str = ", ".join(
            (f"\\textbf{{{MODEL_TEX[m]} {wins[m]}}}" if wins[m] == max_wins
             else f"{MODEL_TEX[m]} {wins[m]}")
            for m in MODEL_ORDER if wins[m] > 0)
        med_best = np.median(best_vals)
        rows.append(
            f"{tributo} & {n_series} & {win_str} & {format_dec(med_best, 2)} \\\\")
    from forecasting.config import styled_table

    tex = styled_table(
        gerado_por="generalization.generalization_table",
        caption="Generaliza\\c{c}\\~ao aos munic\\'ipios baianos com mais de "
        "cem mil habitantes: vit\\'orias por modelo (MASE mediano em $h=12$) e "
        "MASE t\\'ipico do vencedor, por tributo.",
        label="tab:generalizacao",
        colspec="l c L c",
        header=["Tributo", "S\\'eries", "Vit\\'orias por modelo",
                "MASE mediano do vencedor"],
        rows=rows,
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        footnote="``Vit\\'orias'': n\\'umero de munic\\'ipios "
        "em que cada modelo teve o menor MASE mediano em $h=12$; "
        "em negrito, o modelo com mais vit\\'orias.",
        stripe=True,
        size="small",
    )
    out = table_path(cfg, "tab_generalizacao")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out


def generalization_figure(cfg: PipelineConfig) -> Path:
    """fig_generalizacao.pdf: boxplot do MASE (h=12) por modelo no conjunto
    estendido, separado por tributo, com a linha de referencia em MASE=1."""
    import matplotlib.pyplot as plt

    from forecasting.plotting import model_boxplot, save_figure, setup_matplotlib_thesis

    setup_matplotlib_thesis()
    cv = _load_extended_cv(cfg)
    h12 = cv[cv["step"] == 12]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True)
    for ax, tributo in zip(axes, ["IPTU", "ISSQN"]):
        data = [h12[(h12["tributo"] == tributo) & (h12["modelo"] == m)]["scaled_err"].to_numpy()
                for m in MODEL_ORDER]
        model_boxplot(ax, data, MODEL_ORDER)
        ax.set_title(tributo)
        if tributo == "IPTU":
            ax.set_ylabel("MASE ($h = 12$)")
    out = save_figure(fig, "fig_generalizacao", cfg.figures_dir_abs)
    plt.close(fig)
    return out


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Roda a generalizacao (se ainda nao houver cache) e gera tabela e figura."""
    if not (cfg.forecasts_dir / "cv_extended.csv").exists():
        run_generalization(cfg)
    return [generalization_table(cfg), generalization_municipality_table(cfg),
            generalization_figure(cfg)]
