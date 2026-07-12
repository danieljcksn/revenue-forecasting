# -*- coding: utf-8 -*-
"""Figuras e numeros da analise estadual (le out/cv_bahia.csv).

Produz, em out/:
  fig_mapa_bahia.png/.pdf       mapa coropletico: vantagem do Ensemble sobre a
                                projecao oficial, por municipio (p.p. de erro)
  fig_vantagem_populacao.png/.pdf  vantagem por municipio contra populacao (log)
  fig_winrate_faixas.png/.pdf   taxa de vitoria por faixa populacional
  resumo.json                   numeros citados no relatorio

Confronto identico ao do TCC (Secao 5.7): origens de dezembro, soma dos doze
passos, apenas anos-serie com previsao municipal disponivel; erro anual
percentual absoluto de cada lado, vitoria quando o erro do modelo e menor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS / "src"))
OUT = HERE / "out"

import geodata  # noqa: E402  (modulo irmao)

MUNITAX = "#0582FF"
INK, MUTED, BORDER = "#202124", "#5F6368", "#DADCE0"
NEUTRO = "#F1F3F4"
CMAP = LinearSegmentedColormap.from_list(
    "vantagem", ["#B3564A", "#E5B9B2", "#FFFFFF", "#9CC8F7", MUNITAX])
FAIXAS = [(0, 20_000, "até 20 mil"), (20_000, 50_000, "20 a 50 mil"),
          (50_000, 100_000, "50 a 100 mil"), (100_000, None, "acima de 100 mil")]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)


# ---------- dados -----------------------------------------------------------


def confrontos() -> pd.DataFrame:
    """Uma linha por (municipio, tributo, ano, modelo) com os dois erros."""
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


def vantagem_por_municipio(m: pd.DataFrame) -> pd.DataFrame:
    """Vantagem media do Ensemble por municipio: erro da prefeitura menos erro
    do modelo, em pontos percentuais, sobre os anos-serie confrontaveis."""
    ens = m[m["modelo"] == "Ensemble"].copy()
    ens["vantagem"] = ens["erro_pct_prefeitura"] - ens["err_modelo"]
    agg = ens.groupby(["cod_ibge", "municipio"]).agg(
        vantagem=("vantagem", "mean"), confrontos=("vantagem", "size"),
        vitorias=("vence", "sum")).reset_index()
    return agg.merge(geodata.load_population()[["cod_ibge", "populacao"]],
                     on="cod_ibge", how="left")


# ---------- figuras ---------------------------------------------------------


def mapa(adv: pd.DataFrame) -> None:
    polys = geodata.feature_polygons(geodata.load_malha())
    valores = adv.set_index("cod_ibge")["vantagem"]
    # Escala robusta: satura no percentil 80 de |vantagem|, arredondado a um
    # multiplo de 5, para que a vantagem tipica (mediana ~9 p.p.) mostre cor
    # em vez de quase-branco. Outliers alem disso saturam no tom cheio.
    lim = max(10.0, round(float(np.nanpercentile(valores.abs(), 80)) / 5) * 5)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)

    fig, ax = plt.subplots(figsize=(8.2, 9.0))
    com, sem, cores = [], [], []
    for cod, rings in polys.items():
        for ring in rings:
            xy = np.asarray(ring)
            if cod in valores.index:
                com.append(xy)
                cores.append(CMAP(norm(np.clip(valores[cod], -lim, lim))))
            else:
                sem.append(xy)
    # Sem dados: cinza claro com hachura fina, para nao confundir com a
    # vantagem proxima de zero (quase branca) das series avaliadas.
    ax.add_collection(PolyCollection(sem, facecolors="#EDEEF0",
                                     edgecolors="white", linewidths=0.3,
                                     hatch="////", zorder=1))
    ax.add_collection(PolyCollection(com, facecolors=cores,
                                     edgecolors="white", linewidths=0.3,
                                     zorder=2))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.01)

    fig.suptitle("Onde o Ensemble erra menos que a projeção da prefeitura",
                 x=0.5, y=0.965, fontsize=14, color=INK, fontweight="bold")
    ax.set_title("Vantagem no erro anual médio de 2021 a 2025 (IPTU e ISSQN); "
                 "hachurados: municípios\nsem dados aprovados no controle de "
                 "qualidade", fontsize=9, color=MUTED, pad=6)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.040, pad=0.01, aspect=34,
                        extend="both")
    cbar.set_label("← projeção oficial erra menos     "
                   "vantagem do Ensemble (pontos percentuais)     "
                   "Ensemble erra menos →", fontsize=8.5, color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.outline.set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_mapa_bahia.{ext}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("+ fig_mapa_bahia", flush=True)


def vantagem_populacao(adv: pd.DataFrame) -> None:
    d = adv.dropna(subset=["populacao"]).copy()
    # Corte de eixo para nao deixar poucos municipios pequenos, onde a
    # prefeitura erra por larga margem, comprimirem a nuvem principal. Os
    # pontos acima do teto sao desenhados na borda, com anotacao do total.
    teto = max(60.0, round(float(np.nanpercentile(d["vantagem"], 90)) / 10) * 10)
    piso = min(-20.0, round(float(np.nanpercentile(d["vantagem"], 3)) / 10) * 10)
    d["y_plot"] = d["vantagem"].clip(piso, teto)
    acima = int((d["vantagem"] > teto).sum())

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhline(0, color=BORDER, lw=1.0)
    ganha = d["vantagem"] > 0
    ax.scatter(d.loc[ganha, "populacao"], d.loc[ganha, "y_plot"], s=18,
               color=MUNITAX, alpha=0.75, lw=0, label="Ensemble erra menos")
    ax.scatter(d.loc[~ganha, "populacao"], d.loc[~ganha, "y_plot"], s=18,
               color="#B3564A", alpha=0.70, lw=0,
               label="projeção oficial erra menos")

    d["quintil"] = pd.qcut(d["populacao"], 5, duplicates="drop")
    med = d.groupby("quintil", observed=True).agg(
        x=("populacao", "median"), y=("vantagem", "median"))
    ax.plot(med["x"], med["y"], color=INK, lw=1.8, marker="o", ms=4.5,
            label="mediana por quintil de população", zorder=5)

    if acima:
        ax.text(0.015, 0.975, f"↑ {acima} municípios pequenos acima de "
                f"{teto:.0f} p.p.", transform=ax.transAxes, va="top",
                fontsize=7.6, color=MUTED, fontstyle="italic")
    ax.set_xscale("log")
    ax.set_ylim(piso * 1.05, teto * 1.08)
    ax.set_xlabel("População do município (escala log)", fontsize=9, color=MUTED)
    ax.set_ylabel("Vantagem média (p.p. de erro anual)", fontsize=9, color=MUTED)
    ax.set_title("A margem de vantagem tende a crescer onde o município é menor",
                 fontsize=11.5, color=INK, pad=10, fontweight="bold")
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_vantagem_populacao.{ext}", dpi=220,
                    facecolor="white")
    plt.close(fig)
    print("+ fig_vantagem_populacao", flush=True)


def winrate_faixas(m: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """Por faixa populacional: margem mediana de vantagem (barras, que tem
    gradiente com o porte) e taxa de vitoria (rotulo, alta e quase uniforme).

    Combina as duas leituras para nao sugerir um gradiente na frequencia de
    vitoria que os dados nao mostram: vence-se 2/3 das vezes em qualquer porte,
    mas a MARGEM e maior nos municipios menores."""
    ens = m[m["modelo"] == "Ensemble"].merge(
        geodata.load_population()[["cod_ibge", "populacao"]],
        on="cod_ibge", how="left").dropna(subset=["populacao"])
    advp = adv.dropna(subset=["populacao"])
    rows = []
    for lo, hi, rotulo in FAIXAS:
        sel = ens[(ens["populacao"] >= lo)
                  & ((hi is None) | (ens["populacao"] < (hi or np.inf)))]
        selm = advp[(advp["populacao"] >= lo)
                    & ((hi is None) | (advp["populacao"] < (hi or np.inf)))]
        rows.append(dict(faixa=rotulo, n=len(sel),
                         winrate=100 * sel["vence"].mean(),
                         margem_mediana=float(selm["vantagem"].median()),
                         n_munis=len(selm)))
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xs = np.arange(len(tab))
    ax.bar(xs, tab["margem_mediana"], 0.56, color=MUNITAX)
    for x, r in tab.iterrows():
        ax.text(x, r["margem_mediana"] + 0.4,
                f"{r['margem_mediana']:.1f}".replace(".", ",") + " p.p.",
                ha="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(x, 0.4, f"vitória em {r['winrate']:.0f}%".replace(".", ","),
                ha="center", fontsize=7.6, color="white")
    ax.set_xticks(xs, tab["faixa"], fontsize=8.5)
    ax.set_ylim(0, tab["margem_mediana"].max() * 1.2)
    ax.set_ylabel("Margem mediana de vantagem (p.p. de erro anual)",
                  fontsize=9, color=MUTED)
    ax.set_title("A vitória é frequente em todo porte; a margem cresce nos "
                 "menores", fontsize=11.5, color=INK, pad=10, fontweight="bold")
    _style(ax)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_winrate_faixas.{ext}", dpi=220,
                    facecolor="white")
    plt.close(fig)
    print("+ fig_winrate_faixas", flush=True)
    return tab


# ---------- resumo ----------------------------------------------------------


def resumo(m: pd.DataFrame, adv: pd.DataFrame, faixas: pd.DataFrame) -> dict:
    ens = m[m["modelo"] == "Ensemble"]
    n_conf = len(ens)
    expost = 0
    for _k, b in m.groupby(["cod_ibge", "tributo"]):
        best = b.groupby("modelo")["err_modelo"].mean().idxmin()
        expost += int(b[(b["modelo"] == best)]["vence"].sum())
    d = adv.dropna(subset=["populacao"])
    corr = float(np.corrcoef(np.log10(d["populacao"]), d["vantagem"])[0, 1])
    out = dict(
        series=int(m[["cod_ibge", "tributo"]].drop_duplicates().shape[0]),
        municipios=int(m["cod_ibge"].nunique()),
        confrontos=n_conf,
        ens_vitorias=int(ens["vence"].sum()),
        ens_winrate=round(100 * ens["vence"].mean(), 1),
        expost_vitorias=expost,
        expost_winrate=round(100 * expost / n_conf, 1),
        municipios_vantagem_positiva=int((adv["vantagem"] > 0).sum()),
        municipios_total_mapa=int(len(adv)),
        vantagem_mediana_pp=round(float(adv["vantagem"].median()), 1),
        corr_logpop_vantagem=round(corr, 3),
        winrate_por_faixa=[
            {"faixa": r["faixa"], "n": int(r["n"]),
             "winrate": round(r["winrate"], 1),
             "margem_mediana": round(r["margem_mediana"], 1)}
            for r in faixas.to_dict(orient="records")],
    )
    (OUT / "resumo.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    m = confrontos()
    adv = vantagem_por_municipio(m)
    mapa(adv)
    vantagem_populacao(adv)
    faixas = winrate_faixas(m, adv)
    resumo(m, adv, faixas)


if __name__ == "__main__":
    main()
