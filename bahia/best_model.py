# -*- coding: utf-8 -*-
"""Ensemble fixo vs. "melhor modelo": para onde vao os numeros.

Tres estrategias de escolha de modelo, todas confrontadas com a projecao da
prefeitura sob a MESMA regra (origens de dezembro, soma dos doze passos):

1. cada modelo isolado, fixo no estado inteiro (inclui o Ensemble);
2. melhor modelo por serie, ESCOLHIDO COM O RESULTADO NA MAO (ex-post). E o que
   Oliveira (2024) faz: reporta, por serie, o metodo que saiu melhor. Usa o
   futuro na selecao, entao e um TETO otimista, nao uma estrategia operavel;
3. melhor modelo por serie, escolhido so com o passado (walk-forward): no ano
   Y, adota o modelo de menor erro anual medio nos anos < Y da propria serie;
   no primeiro ano, sem historico, cai no Ensemble. Deployable, sem vazamento.

Escreve out/best_model.json e imprime a tabela.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
OUT = HERE / "out"

FORMAL = ["ETS", "SARIMA", "Theta", "Prophet"]
ALL_MODELS = ["Naive"] + FORMAL + ["Ensemble"]


def load_confrontos() -> pd.DataFrame:
    from forecasting.config import load_config
    from forecasting.io import load_prefeitura_forecast

    cv = pd.read_csv(OUT / "cv_bahia.csv")
    cv["err_modelo"] = (100 * (cv["pred_annual"] - cv["real_annual"]).abs()
                        / cv["real_annual"].abs())
    cfg = load_config()
    pf = load_prefeitura_forecast(cfg).rename(columns={"year": "target_year"})
    pf = pf[["cod_ibge", "tributo", "target_year", "erro_pct_prefeitura"]]
    m = cv.merge(pf, on=["cod_ibge", "tributo", "target_year"], how="inner")
    m["vence"] = m["err_modelo"] < m["erro_pct_prefeitura"]
    return m


def winrate(sub: pd.DataFrame) -> tuple[int, int, float]:
    v, n = int(sub["vence"].sum()), len(sub)
    return v, n, round(100 * v / n, 1)


def escolha_expost(m: pd.DataFrame) -> pd.DataFrame:
    """Por serie, o modelo de menor erro anual medio (usa todos os anos)."""
    best = (m.groupby(["cod_ibge", "tributo", "modelo"])["err_modelo"].mean()
            .reset_index()
            .sort_values("err_modelo")
            .drop_duplicates(["cod_ibge", "tributo"])[
                ["cod_ibge", "tributo", "modelo"]]
            .rename(columns={"modelo": "escolhido"}))
    sel = m.merge(best, on=["cod_ibge", "tributo"])
    return sel[sel["modelo"] == sel["escolhido"]]


def escolha_walkforward(m: pd.DataFrame) -> pd.DataFrame:
    """Por serie e ano Y, o modelo de menor erro medio nos anos < Y da propria
    serie; sem historico (primeiro ano), cai no Ensemble. Sem vazamento."""
    linhas = []
    for (cod, trib), bloco in m.groupby(["cod_ibge", "tributo"]):
        anos = sorted(bloco["target_year"].unique())
        for y in anos:
            passado = bloco[bloco["target_year"] < y]
            if passado.empty:
                escolhido = "Ensemble"
            else:
                escolhido = (passado.groupby("modelo")["err_modelo"].mean()
                             .idxmin())
            linha = bloco[(bloco["target_year"] == y)
                          & (bloco["modelo"] == escolhido)]
            linhas.append(linha)
    return pd.concat(linhas, ignore_index=True)


def figura(resultado: dict) -> None:
    """Barras comparando as estrategias de escolha de modelo. A barra ex-post
    (com hindsight) e hachurada, para marcar que nao e operavel."""
    import matplotlib.pyplot as plt

    MUNITAX, INK, MUTED, BORDER = "#0582FF", "#202124", "#5F6368", "#DADCE0"
    fx = resultado["modelos_fixos"]
    es = resultado["estrategias"]
    linhas = [
        ("Melhor por série, com o resultado\nna mão (ex-post, estilo Oliveira)",
         es["melhor_expost"]["taxa"], "hindsight"),
        ("Naïve sazonal fixo", fx["Naive"]["taxa"], "fixo"),
        ("Ensemble fixo (adotado no TCC)", fx["Ensemble"]["taxa"], "adotado"),
        ("Melhor por série, só com o\npassado (walk-forward, operável)",
         es["melhor_walkforward"]["taxa"], "deploy"),
    ]
    cores = {"hindsight": "#C9CDD2", "fixo": "#9AA0A6",
             "adotado": MUNITAX, "deploy": "#7CB7F5"}
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ys = np.arange(len(linhas))[::-1]
    for y, (rot, tx, tipo) in zip(ys, linhas):
        kw = dict(color=cores[tipo], height=0.62)
        if tipo == "hindsight":
            kw.update(hatch="////", edgecolor="#8A8F96", color="#EDEEF0")
        ax.barh(y, tx, **kw)
        ax.text(tx + 0.6, y, f"{tx:.1f}%".replace(".", ","), va="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.axvline(fx["Ensemble"]["taxa"], color=MUTED, lw=1.0, ls=(0, (4, 3)),
               zorder=0)
    ax.set_yticks(ys, [l[0] for l in linhas], fontsize=8.6)
    ax.set_xlim(0, 82)
    ax.set_xlabel("Anos-série em que vence a projeção da prefeitura (%)",
                  fontsize=9, color=MUTED)
    ax.set_title("O ganho do “melhor modelo” é uma miragem de hindsight",
                 fontsize=12, color=INK, pad=10, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_melhor_modelo.{ext}", dpi=220,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("+ fig_melhor_modelo")


def main() -> None:
    m = load_confrontos()
    total = m[m["modelo"] == "Ensemble"].shape[0]
    print(f"Confrontos por serie-ano: {total}\n")
    print(f"{'estrategia':32s} {'vence':>12s} {'taxa':>7s}")
    print("-" * 54)

    resultado = {"confrontos": total, "modelos_fixos": {}, "estrategias": {}}

    for mod in ALL_MODELS:
        v, n, tx = winrate(m[m["modelo"] == mod])
        marca = "  <- Ensemble" if mod == "Ensemble" else ""
        print(f"{'fixo: ' + mod:32s} {f'{v}/{n}':>12s} {tx:>6.1f}%{marca}")
        resultado["modelos_fixos"][mod] = dict(vitorias=v, n=n, taxa=tx)

    print()
    exp = escolha_expost(m)
    v, n, tx = winrate(exp)
    print(f"{'melhor por serie (ex-post)':32s} {f'{v}/{n}':>12s} {tx:>6.1f}%"
          f"  <- estilo Oliveira (usa o futuro)")
    resultado["estrategias"]["melhor_expost"] = dict(vitorias=v, n=n, taxa=tx)

    wf = escolha_walkforward(m)
    v, n, tx = winrate(wf)
    print(f"{'melhor por serie (walk-forward)':32s} {f'{v}/{n}':>12s} "
          f"{tx:>6.1f}%  <- deployable, sem vazamento")
    resultado["estrategias"]["melhor_walkforward"] = dict(vitorias=v, n=n,
                                                          taxa=tx)

    # Tetos teoricos (usam o futuro; nao operaveis, so referencia).
    maxwins = int(m.groupby(["cod_ibge", "tributo", "modelo"])["vence"].sum()
                  .reset_index().sort_values("vence", ascending=False)
                  .drop_duplicates(["cod_ibge", "tributo"])["vence"].sum())
    print(f"{'melhor por serie (maximiza vit.)':32s} {f'{maxwins}/{total}':>12s} "
          f"{100*maxwins/total:>6.1f}%  <- teto, muda criterio")
    resultado["estrategias"]["melhor_maxwins"] = dict(
        vitorias=maxwins, n=total, taxa=round(100 * maxwins / total, 1))
    orac = int(m.groupby(["cod_ibge", "tributo", "target_year"])["vence"]
               .max().sum())
    print(f"{'oraculo por ano':32s} {f'{orac}/{total}':>12s} "
          f"{100*orac/total:>6.1f}%  <- teto absoluto (troca a cada ano)")
    resultado["oraculo_ano"] = dict(vitorias=orac, n=total,
                                    taxa=round(100 * orac / total, 1))

    # Quantas series o ex-post escolhe cada modelo (diversidade do vencedor).
    dist = (exp.drop_duplicates(["cod_ibge", "tributo"])["modelo"]
            .value_counts().to_dict())
    resultado["expost_escolhas"] = {k: int(v) for k, v in dist.items()}
    print("\nDistribuicao do vencedor ex-post por serie:")
    for k, val in dist.items():
        print(f"  {k:10s} {val}")

    (OUT / "best_model.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    figura(resultado)
    print(f"\n-> {OUT/'best_model.json'}")


if __name__ == "__main__":
    main()
