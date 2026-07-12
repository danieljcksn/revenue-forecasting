"""Orquestra a geracao de TODAS as tabelas e figuras do TCC.

Uso:
    python scripts/build_tex_artifacts.py --all
    python scripts/build_tex_artifacts.py --eda           # Secao 5.1 + Cap. 4
    python scripts/build_tex_artifacts.py --models        # Secao 5.2
    python scripts/build_tex_artifacts.py --evaluation    # Secoes 5.3-5.4 + Cap. 6
    python scripts/build_tex_artifacts.py --benchmarks    # Secao 5.6
    python scripts/build_tex_artifacts.py --generalizacao # Secao 5.7

Le o cache de previsoes (data/forecasts/, versionado; ver RUN_ORDER.md) e
regenera as tabelas e figuras vivas do documento em tcc-latex/{tables,figures}/
generated/. ATENCAO: --eda recomputa os testes da tab_estacionariedade e
--models re-fita as tabelas de parametros (sensiveis a versao de biblioteca);
os demais passos leem apenas o cache.
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
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Executar todas as etapas em ordem.")
    parser.add_argument("--eda", action="store_true",
                        help="Secao 5.1 + Cap. 4 - descritivas, estacionariedade, "
                             "painel de series, transformacao log, STL.")
    parser.add_argument("--models", action="store_true",
                        help="Secao 5.2 - tabelas de parametros (re-fit) e "
                             "figura de previsoes.")
    parser.add_argument("--evaluation", action="store_true",
                        help="Secoes 5.3-5.4 + Cap. 6 - tabela consolidada, "
                             "heatmap, curvas de horizonte e de ano-alvo.")
    parser.add_argument("--benchmarks", action="store_true",
                        help="Secao 5.6 - confronto com prefeitura (tabela + "
                             "dumbbell) e Oliveira (2024).")
    parser.add_argument("--generalizacao", action="store_true",
                        help="Secao 5.7 - conjunto ampliado (tabelas, barras e "
                             "dispersao vs prefeitura).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path para .tcc-pipeline.json (default: busca na hierarquia).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Pipeline config carregado de TCC root: {cfg.tcc_root}")
    print(f"  Tabelas em : {cfg.tables_dir_abs}")
    print(f"  Figuras em : {cfg.figures_dir_abs}")

    failures = 0
    if args.all or args.eda:
        failures += _step("EDA (Secao 5.1 + Cap. 4)", eda.run_all, cfg)
    if args.all or args.models:
        # O passo de modelos re-fita ETS/Theta via statsforecast (extra
        # opcional "precisao"). Sem a lib, avisa como rodar em vez de so
        # estourar ImportError.
        try:
            import statsforecast  # noqa: F401
            failures += _step("Modelos (Secao 5.2)", models.run_all, cfg)
        except ImportError:
            print("\n=== Modelos (Secao 5.2) ===")
            print("  ! PULADO: requer 'statsforecast' (extra 'precisao').")
            print("    Instale com: pip install -r requirements-sf-lock.txt")
            print("    ou rode este passo no venv dedicado: "
                  "python scripts/build_tex_artifacts.py --models")
            failures += 1
    if args.all or args.evaluation:
        failures += _step("Avaliacao (Secoes 5.3-5.4 + Cap. 6)", evaluation.run_all, cfg)
    if args.all or args.benchmarks:
        failures += _step("Benchmarks (Secao 5.6)", benchmarks.run_all, cfg)
    if args.all or args.generalizacao:
        from forecasting import generalization
        failures += _step("Generalizacao (Secao 5.7)", generalization.run_all, cfg)

    if not any([args.all, args.eda, args.models, args.evaluation,
                args.benchmarks, args.generalizacao]):
        parser.print_help()
        return 1

    print(f"\nConcluido. {failures} passo(s) com erro.")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
