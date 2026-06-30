"""Orquestra a geracao de TODAS as tabelas e figuras do TCC.

Equivalente a "rodar todos os notebooks de cima a baixo, mas via Python puro".

Uso:
    python scripts/build_tex_artifacts.py --all
    python scripts/build_tex_artifacts.py --eda          # so Cap 5.1
    python scripts/build_tex_artifacts.py --models       # so Cap 5.2
    python scripts/build_tex_artifacts.py --evaluation   # so Cap 5.3
    python scripts/build_tex_artifacts.py --benchmarks   # Cap 5.4
    python scripts/build_tex_artifacts.py --generalizacao # Cap 5.6

Le o cache de previsoes (data/forecasts/, produzido por run_pipeline.py) e
regenera todas as tabelas e figuras do TCC em tcc-latex/{tables,figures}/
generated/. As funcoes `run_all` de cada modulo estao implementadas; este
script apenas as orquestra na ordem correta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from forecasting import benchmarks, eda, evaluation, models
from forecasting.config import load_config


def _step(label: str, fn: Callable, cfg) -> int:
    print(f"\n=== {label} ===")
    try:
        artifacts = fn(cfg)
        for path in artifacts or []:
            print(f"  + {path}")
        return 0
    except Exception as e:
        print(f"  ! ERROR: {type(e).__name__}: {e}")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Executar todas as etapas em ordem.")
    parser.add_argument("--eda", action="store_true",
                        help="Cap 5.1 - estatisticas descritivas, decomposicao, ACF/PACF.")
    parser.add_argument("--models", action="store_true",
                        help="Cap 5.2 - treinamento dos quatro modelos.")
    parser.add_argument("--evaluation", action="store_true",
                        help="Cap 5.3 - metricas e ranque por municipio.")
    parser.add_argument("--benchmarks", action="store_true",
                        help="Cap 5.4 - confronto com prefeitura e Oliveira (2024).")
    parser.add_argument("--generalizacao", action="store_true",
                        help="Cap 5.6 - generalizacao aos municipios baianos populosos.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path para .tcc-pipeline.json (default: busca na hierarquia).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Pipeline config carregado de TCC root: {cfg.tcc_root}")
    print(f"  Tabelas em : {cfg.tables_dir_abs}")
    print(f"  Figuras em : {cfg.figures_dir_abs}")

    failures = 0
    if args.all or args.eda:
        failures += _step("EDA (Cap 5.1)", eda.run_all, cfg)
    if args.all or args.models:
        failures += _step("Modelos (Cap 5.2)", models.run_all, cfg)
    if args.all or args.evaluation:
        failures += _step("Avaliacao (Cap 5.3)", evaluation.run_all, cfg)
    if args.all or args.benchmarks:
        failures += _step("Benchmarks (Cap 5.4)", benchmarks.run_all, cfg)
    if args.all or args.generalizacao:
        from forecasting import generalization
        failures += _step("Generalizacao (Cap 5.6)", generalization.run_all, cfg)

    if not any([args.all, args.eda, args.models, args.evaluation,
                args.benchmarks, args.generalizacao]):
        parser.print_help()
        return 1

    print(f"\nConcluido. {failures} passo(s) com erro/stub.")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
