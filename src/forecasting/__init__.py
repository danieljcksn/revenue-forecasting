"""Pacote `forecasting` - infraestrutura de analise do TCC.

Escopo enxuto, alinhado a um TCC de graduacao defensavel em uma sessao
de banca:

- 3 municipios baianos em profundidade (Salvador, Camacari, Ilheus) e
  generalizacao aos 18 municipios com mais de cem mil habitantes
- 2 tributos proprios (IPTU, ISSQN)
- 6 previsores: Naive sazonal, ETS, SARIMA, Prophet, Theta e Ensemble
  (portfolio canonico cacheado em cv_all.csv; ver RUN_ORDER.md)
- 2 horizontes: h=1 (mes) e h=12 (ano - LOA)
- Metricas: MAE, MAPE, MASE
- Avaliacao: validacao por origem movel (sem teste DM)
- Comparacoes: vs prefeitura (RREO-Anexo 03, Previsao Atualizada) e vs
  Oliveira (2024)

Modulos:
    config          carregamento de .tcc-pipeline.json, constantes,
                    formatadores BR e o estilo unico de tabelas (styled_table)
    io              leitura/escrita de artefatos do TCC
    plotting        estilo visual padronizado das figuras
    eda             analise exploratoria (artefatos da Secao 5.1 + Cap. 4)
    models          ajustadores do nucleo e validacao por origem movel
    model_reports   tabelas de parametros e figura de previsoes (Secao 5.2)
    evaluation      metricas, heatmap, curvas de horizonte/ano-alvo
    benchmarks      confronto com prefeitura e Oliveira (2024)
    generalization  conjunto ampliado (Secao 5.7)

Os notebooks em ``notebooks/`` sao clientes finos destes modulos.
Este projeto consome o pacote `siconfi-collector` como dependencia.
"""

from forecasting.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
