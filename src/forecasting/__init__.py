"""Pacote `forecasting` - infraestrutura de analise do trabalho.

Escopo empirico principal:

- 3 municipios baianos (Salvador, Camacari, Ilheus)
- 2 tributos proprios (IPTU, ISSQN)
- 6 previsores reportados: Naive Sazonal, ETS, SARIMA, Prophet, Theta, Ensemble
- 2 horizontes: h=1 (mes) e h=12 (ano - LOA)
- Metricas: MAE, MAPE, MASE
- Avaliacao: rolling origin (sem teste DM)
- Comparacoes: vs prefeitura (RREO-Anexo 01) e vs Oliveira (2024)

O driver historico em ``scripts/run_pipeline.py`` reproduz o nucleo original de
quatro modelos; os artefatos finais usam o cache canonico em ``data/forecasts/``.

Modulos:
    config       carregamento de .tcc-pipeline.json, constantes e formatadores
    io           leitura/escrita de artefatos do TCC
    plotting     estilo visual padronizado
    eda          analise exploratoria (gera artefatos do Cap 5.1)
    models       wrappers de treinamento dos quatro modelos
    evaluation   metricas e ranque por municipio
    benchmarks   comparacao com prefeitura e Oliveira (2024)

Os notebooks em ``notebooks/`` sao clientes finos destes modulos.
Este projeto consome o pacote `siconfi-collector` como dependencia.
"""

from forecasting.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
