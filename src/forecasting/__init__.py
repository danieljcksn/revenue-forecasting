"""Forecasting analysis package.

Main empirical scope:

- 3 municipios baianos (Salvador, Camacari, Ilheus)
- 2 tributos proprios (IPTU, ISSQN)
- 6 previsores reportados: Naive Sazonal, ETS, SARIMA, Prophet, Theta, Ensemble
- 2 horizontes: h=1 (mes) e h=12 (ano - LOA)
- Metricas: MAE, MAPE, MASE
- Avaliacao: rolling origin (sem teste DM)
- Comparacoes: vs prefeitura (RREO-Anexo 01) e vs Oliveira (2024)

O driver historico em ``scripts/run_pipeline.py`` reproduz o nucleo original de
quatro modelos; os artefatos finais usam o cache canonico em ``data/forecasts/``.

Modules:
    config       configuration, constants, and formatters
    io           forecast-cache and artifact IO
    plotting     shared visual style
    eda          exploratory analysis and series preparation
    models       model training wrappers
    evaluation   metrics and model rankings
    benchmarks   external benchmark comparisons

Os notebooks em ``notebooks/`` sao clientes finos destes modulos.
Este projeto consome o pacote `siconfi-collector` como dependencia.
"""

from forecasting.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
