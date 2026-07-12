# -*- coding: utf-8 -*-
"""Placar das variantes experimentais (proveniencia do paragrafo do Cap. 6).

O texto afirma: "a melhor delas leva o MASE mediano de 0,91 a 0,84; em troca,
cedem no confronto anual com a prefeitura, caindo de vinte e quatro para
dezenove vitorias em trinta". Este script recomputa o placar completo a partir
dos caches versionados em data/forecasts/experiments/ (variantes re-ajustadas
com janela deslizante w72, log, damped etc.) + cv_all.csv, e grava
data/reports/variantes.md identificando a variante da frase:
Ens5 w72+Naive (media de ETS/SARIMA/Theta com janela deslizante de 72 meses,
Prophet e Naive).

Uso:  python scripts/report_variantes.py     (venv principal; le so caches)
"""

from __future__ import annotations

import sys

import pandas as pd

from forecasting.config import load_config
from forecasting.io import load_prefeitura_forecast

F4 = ["ETS", "SARIMA", "Theta", "Prophet"]
INDIVIDUAIS = ["Naive", "ETS", "SARIMA", "Theta", "Prophet", "Ensemble",
               "NaiveDrift", "ETS_damped", "ETS_log", "Theta_log",
               "ETS_w72", "SARIMA_w72", "Theta_w72"]
COMBOS = {
    "Ens4 canonico": F4,
    "Ens5 +Naive": F4 + ["Naive"],
    "Ens5 +NaiveDrift": F4 + ["NaiveDrift"],
    "Ens4 w72": ["ETS_w72", "SARIMA_w72", "Theta_w72", "Prophet"],
    "Ens5 w72+Naive": ["ETS_w72", "SARIMA_w72", "Theta_w72", "Prophet", "Naive"],
    "Ens4 ETS_log": ["ETS_log", "SARIMA", "Theta", "Prophet"],
    "Ens4 logs": ["ETS_log", "SARIMA", "Theta_log", "Prophet"],
    "Ens5 logs+Naive": ["ETS_log", "SARIMA", "Theta_log", "Prophet", "Naive"],
    "Ens3 ETS+Theta+Naive": ["ETS", "Theta", "Naive"],
    "Ens2 ETS+Naive": ["ETS", "Naive"],
}


def main() -> int:
    cfg = load_config()
    exp_dir = cfg.forecasts_dir / "experiments"
    cv = pd.read_csv(cfg.forecasts_dir / "cv_all.csv",
                     parse_dates=["origin", "target_date"])
    base_cols = ["municipio", "tributo", "origin", "step", "y_true", "y_pred",
                 "insample_scale", "modelo"]
    frames = [cv[base_cols]]
    for f in sorted(exp_dir.glob("cv_exp_*.csv")):
        d = pd.read_csv(f, parse_dates=["origin", "target_date"])
        d["modelo"] = f.stem.replace("cv_exp_", "")
        frames.append(d[base_cols])
    allcv = pd.concat(frames, ignore_index=True)

    wide = allcv.pivot_table(index=["municipio", "tributo", "origin", "step"],
                             columns="modelo", values="y_pred").reset_index()
    meta = allcv.drop_duplicates(["municipio", "tributo", "origin", "step"])[
        ["municipio", "tributo", "origin", "step", "y_true", "insample_scale"]]
    wide = wide.merge(meta, on=["municipio", "tributo", "origin", "step"])

    pf = load_prefeitura_forecast(cfg)
    code_to_key = {m.cod_ibge: k for k, m in cfg.municipalities.items()}
    pf = pf[pf["cod_ibge"].isin(code_to_key)].copy()
    pf["municipio"] = pf["cod_ibge"].map(code_to_key)
    pf = pf.rename(columns={"year": "target_year"})[
        ["municipio", "tributo", "target_year", "erro_pct_prefeitura"]]

    def score(pred, name):
        d = wide.copy()
        d["y_pred"] = pred
        d = d[d["y_pred"].notna()]
        d["scaled"] = (d["y_true"] - d["y_pred"]).abs() / d["insample_scale"]
        m1 = float(d.loc[d["step"] == 1, "scaled"].median())
        m12 = float(d.loc[d["step"] == 12, "scaled"].median())
        dec = d[(d["origin"].dt.month == 12) & d["step"].between(1, 12)].copy()
        dec["target_year"] = dec["origin"].dt.year + 1
        g = dec.groupby(["municipio", "tributo", "target_year"]).agg(
            pred=("y_pred", "sum"), real=("y_true", "sum"),
            n=("step", "count")).reset_index()
        g = g[g["n"] == 12]
        g["err"] = 100 * (g["pred"] - g["real"]).abs() / g["real"].abs()
        m = g.merge(pf, on=["municipio", "tributo", "target_year"], how="inner")
        wins = int((m["err"] < m["erro_pct_prefeitura"]).sum())
        return dict(nome=name, h1=m1, h12=m12, wins=wins, n=len(m),
                    err=float(m["err"].mean()))

    rows = []
    for mod in INDIVIDUAIS:
        if mod in wide.columns:
            rows.append(score(wide[mod], mod))
    for name, members in COMBOS.items():
        if all(m in wide.columns for m in members):
            rows.append(score(wide[members].mean(axis=1), name))
    rows.append(score(wide[F4 + ["NaiveDrift"]].median(axis=1),
                      "Ens5 +NaiveDrift (mediana)"))
    rows.append(score(wide[F4 + ["Naive"]].median(axis=1),
                      "Ens5 +Naive (mediana)"))

    by_name = {r["nome"]: r for r in rows}
    canon = by_name["Ensemble"]
    reativa = by_name["Ens5 w72+Naive"]

    # ---- tabela LaTeX do apendice (tab_variantes.tex, casa de estilo) ----
    from forecasting.config import format_dec, styled_table
    from forecasting.io import table_path
    NOMES_TEX = {
        "Naive": "\\textit{Na\\\"ive}", "ETS": "ETS", "SARIMA": "SARIMA",
        "Theta": "Theta", "Prophet": "Prophet",
        "Ensemble": "\\emph{Ensemble} (can\\^onico)",
        "NaiveDrift": "\\textit{Na\\\"ive} com deriva",
        "ETS_damped": "ETS amortecido", "ETS_log": "ETS log",
        "Theta_log": "Theta log", "ETS_w72": "ETS w72",
        "SARIMA_w72": "SARIMA w72", "Theta_w72": "Theta w72",
        "Ens4 canonico": None,  # igual ao Ensemble canonico; omitir
        "Ens5 +Naive": "Ens5 +\\textit{Na\\\"ive}",
        "Ens5 +NaiveDrift": "Ens5 +\\textit{Na\\\"ive} deriva",
        "Ens4 w72": "Ens4 w72",
        "Ens5 w72+Naive": "Ens5 w72+\\textit{Na\\\"ive}",
        "Ens4 ETS_log": "Ens4 ETS log", "Ens4 logs": "Ens4 logs",
        "Ens5 logs+Naive": "Ens5 logs+\\textit{Na\\\"ive}",
        "Ens3 ETS+Theta+Naive": "Ens3 ETS+Theta+\\textit{Na\\\"ive}",
        "Ens2 ETS+Naive": "Ens2 ETS+\\textit{Na\\\"ive}",
        "Ens5 +NaiveDrift (mediana)": "Ens5 +\\textit{Na\\\"ive} deriva (mediana)",
        "Ens5 +Naive (mediana)": "Ens5 +\\textit{Na\\\"ive} (mediana)",
    }
    rows_tex = []
    for r in rows:
        nome = NOMES_TEX.get(r["nome"], r["nome"])
        if nome is None:
            continue
        rows_tex.append(
            f"{nome} & {format_dec(r['h1'], 3)} & {format_dec(r['h12'], 3)} & "
            f"{r['wins']}/{r['n']} \\\\")
    tex = styled_table(
        gerado_por="scripts/report_variantes.py",
        caption="Bateria de variantes sob o mesmo protocolo",
        label="tab:variantes",
        colspec="L C C C",
        header=["Configura\\c{c}\\~ao", "MASE $h=1$", "MASE $h=12$",
                "Vit\\'orias anuais"],
        rows=rows_tex,
        footnote=("MASE mediano das dobras do n\\'ucleo por horizonte; "
                  "vit\\'orias anuais: anos-s\\'erie, dos trinta do n\\'ucleo, em "
                  "que a configura\\c{c}\\~ao supera a Previs\\~ao Inicial da "
                  "prefeitura no total do exerc\\'icio. Sufixo w72: janela "
                  "deslizante de 72 meses; log: ajuste sobre o logaritmo; "
                  "(mediana): combina\\c{c}\\~ao pela mediana em vez da m\\'edia. "
                  "Ens$k$ indica o n\\'umero de previsores combinados."),
        fonte="Elabora\\c{c}\\~ao pr\\'opria.",
        stripe=True,
        size="scriptsize",
        arraystretch="1.15",
        floating=False,
    )
    ttex = table_path(cfg, "tab_variantes")
    ttex.write_text(tex, encoding="utf-8")
    print(f"gravado: {ttex}")

    out = cfg.analysis_root / "data" / "reports" / "variantes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bateria de variantes: proveniencia do paragrafo do Cap. 6",
        "",
        "> Gerado por `scripts/report_variantes.py` a partir de",
        "> `data/forecasts/experiments/cv_exp_*.csv` (variantes re-ajustadas sob",
        "> o mesmo protocolo) e do cache canonico. Sufixos: w72 = janela",
        "> DESLIZANTE de 72 meses (reage mais depressa ao degrau de nivel);",
        "> log = ajuste sobre o logaritmo; damped = tendencia amortecida.",
        "",
        "## A frase do texto",
        "",
        f"A variante da frase e a **Ens5 w72+Naive**: MASE mediano h=12 de",
        f"{canon['h12']:.2f} (Ensemble canonico) para {reativa['h12']:.2f}, e",
        f"vitorias anuais de {canon['wins']}/{canon['n']} para "
        f"{reativa['wins']}/{reativa['n']}.",
        "Reagir ao degrau melhora o decimo segundo mes e custa o total anual.",
        "",
        "## Placar completo",
        "",
        "| Variante | MASE h=1 | MASE h=12 | Vitorias anuais | Erro anual medio (%) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['nome']} | {r['h1']:.3f} | {r['h12']:.3f} | "
                     f"{r['wins']}/{r['n']} | {r['err']:.1f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"gravado: {out}")
    print(f"conferencia Cap.6: {canon['h12']:.2f} -> {reativa['h12']:.2f}; "
          f"{canon['wins']}/{canon['n']} -> {reativa['wins']}/{reativa['n']}")
    ok = (round(canon["h12"], 2) == 0.91 and round(reativa["h12"], 2) == 0.84
          and canon["wins"] == 24 and reativa["wins"] == 19)
    print("OK: bate com a prosa." if ok else "DIVERGE da prosa!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
