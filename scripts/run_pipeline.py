"""Driver da modelagem: validacao por origem movel das seis series x quatro modelos.

Executa, para cada serie (municipio x tributo) e cada modelo parcimonioso
(Naive Sazonal, ETS, SARIMA, Prophet), a validacao cruzada por origem movel
descrita na Secao 4.7 do TCC, e cacheia o resultado consolidado em disco. As
tabelas e figuras dos Capitulos 5 e 6 sao construidas a jusante a partir desse
cache -- de modo que iterar no texto nao exige re-treinar nada.

Saidas (em cfg.forecasts_dir):
  - cv_all.csv         : tabela longa (serie x modelo x origem x passo)
  - params_full.csv    : parametros do ajuste de cada modelo na serie completa
  - run_meta.json      : metadados da execucao (tempos, contagens)

NOTA (reprodutibilidade): o portfolio reportado no TCC tem SEIS modelos (os quatro
acima mais Theta e o Ensemble), com configuracoes corrigidas (AutoETS de taxonomia
completa, SARIMA com D=1 forcado, Prophet mensal sem feriados e Fourier=6), montado
pelos scripts de ``_precisao_run/`` e cacheado em ``cv_all.csv``. Este driver reproduz
o NUCLEO de quatro modelos; consolidar o caminho de seis modelos como pipeline
canonico e tarefa pendente (ver RUN_ORDER.md).

Uso:
  python scripts/run_pipeline.py
"""

from __future__ import annotations

# ruff: noqa: E402
# As variaveis de ambiente de threads BLAS precisam ser definidas ANTES de
# importar numpy/pandas, o que torna os imports abaixo intencionalmente
# posteriores ao bloco de configuracao.
import os

# Evita oversubscricao de threads BLAS quando se fazem muitos ajustes pequenos
# em sequencia (mais rapido com 1 thread por processo).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from forecasting import models as M
from forecasting.config import PipelineConfig, load_config
from forecasting.eda import prepare_series

warnings.filterwarnings("ignore")

def run_models(cfg: PipelineConfig) -> list[Path]:
    # Reprodutibilidade: fixa as seeds globais (numpy/random) ANTES de qualquer
    # ajuste, com a mesma semente dos notebooks. Naive/ETS/SARIMA sao
    # deterministas; isto torna EXPLICITA a semente sob a qual o Prophet e
    # ajustado (Prophet usa otimizacao MAP, deterministica, sem RNG do numpy --
    # ver nota de seeds no resumo / estudos/verificacao/rolling_seeds_out.txt).
    M.set_global_seeds()
    series = prepare_series(cfg, impute=True)
    cv_frames: list[pd.DataFrame] = []
    param_rows: list[dict] = []
    meta: dict = {"fits": {}, "series": {}}

    for (mun_key, tributo), s in series.items():
        mun_name = cfg.municipalities[mun_key].name
        meta["series"][f"{mun_key}-{tributo}"] = int(len(s))

        fitters = M.make_fitters(s)

        for model_name, fit_fn in fitters.items():
            t0 = time.perf_counter()
            cv = M.rolling_origin_cv(
                s, fit_fn, initial_window=72, max_horizon=12, step=1,
            )
            cv.insert(0, "modelo", model_name)
            cv.insert(0, "tributo", tributo)
            cv.insert(0, "municipio", mun_key)
            cv.insert(0, "municipio_nome", mun_name)
            cv["regime"] = cv["target_date"].apply(lambda d: M.covid_regime(d, cfg))
            cv_frames.append(cv)

            # ajuste canonico na serie completa -> tabela de parametros
            full = M.fit_sarima(s) if model_name == "SARIMA" else fit_fn(s)
            param_rows.append({
                "municipio": mun_key, "municipio_nome": mun_name,
                "tributo": tributo, "modelo": model_name,
                "aic": full.aic,
                **{k: v for k, v in full.params.items()
                   if k not in ("last_train_date", "kind")},
            })
            dt = time.perf_counter() - t0
            meta["fits"][f"{mun_key}-{tributo}-{model_name}"] = round(dt, 1)
            print(f"[ok] {mun_name:9s} {tributo:5s} {model_name:8s} "
                  f"origins={cv['origin'].nunique():2d} rows={len(cv):3d} {dt:6.1f}s",
                  flush=True)

    cv_all = pd.concat(cv_frames, ignore_index=True)
    params = pd.DataFrame(param_rows)

    out_dir = cfg.forecasts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p_cv = out_dir / "cv_all.csv"
    p_par = out_dir / "params_full.csv"
    p_meta = out_dir / "run_meta.json"
    cv_all.to_csv(p_cv, index=False, encoding="utf-8")
    params.to_csv(p_par, index=False, encoding="utf-8")
    p_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nGravado: {p_cv} ({len(cv_all)} linhas)")
    print(f"Gravado: {p_par} ({len(params)} linhas)")
    return [p_cv, p_par, p_meta]


if __name__ == "__main__":
    cfg = load_config()
    t0 = time.perf_counter()
    run_models(cfg)
    print(f"\nTempo total: {(time.perf_counter() - t0) / 60:.1f} min")
