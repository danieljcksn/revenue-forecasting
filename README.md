# tcc-forecasting (analysis/)

> **Nota:** este repositório espelha o diretório `analysis/` do monorepo do TCC.
> As referências a `../tcc-latex/`, `../siconfi-collector/` e ao `.tcc-pipeline.json`
> da raiz dizem respeito àquele layout; os dados de entrada vêm do
> [siconfi-collector](https://github.com/danieljcksn/siconfi-collector).

Projeto de análise do TCC de **Daniel Jackson Cavalcante Costa** (UESC, 2026).
Compara seis previsores de séries temporais (Naïve sazonal como baseline, ETS,
SARIMA, Prophet, Theta e Ensemble por média simples) sobre receitas tributárias
municipais (IPTU, ISSQN) em três municípios baianos: Salvador, Camaçari e Ilhéus.

Escopo:

- 3 municípios × 2 tributos = **6 séries** (núcleo); extensões para 31 séries
  (conjunto ampliado) e 230 municípios da Bahia (`bahia/`)
- 6 previsores (portfólio canônico em cache versionado; ver `RUN_ORDER.md` na raiz)
- 2 horizontes: h = 1 (mês) e h = 12 (ano, alvo da LOA)
- Métricas: MASE (principal), MAPE, MAE; avaliação por origem móvel
- Comparações externas: previsão da própria prefeitura (P1, próxima da LOA)
  e Oliveira (2024)

## Separação de responsabilidades

Este projeto é **cliente** do `siconfi-collector`. No monorepo:

```
final-paperz/
├── .tcc-pipeline.json    # config canônico (paths relativos à raiz do repo)
├── tcc-latex/            # o documento (LaTeX)
├── siconfi-collector/    # coleta da API SICONFI + transformações primárias
└── analysis/             # ESTE projeto: modelagem e geração de artefatos
```

- **siconfi-collector**: coleta do RREO-Anexo 03 e transformações primárias
  (série mensal por tributo; previsão da prefeitura P1 cruzada com o realizado).
- **analysis**: EDA, modelagem, avaliação, benchmarks e geração dos artefatos
  LaTeX consumidos pelo documento.

## Layout

```
analysis/
├── pyproject.toml            # deps principais + extra opcional "precisao" (statsforecast)
├── requirements-lock.txt     # env principal pinado (prophet, statsmodels, pmdarima)
├── requirements-sf-lock.txt  # env do statsforecast pinado (AutoETS/AutoTheta)
├── src/forecasting/
│   ├── config.py             # carrega .tcc-pipeline.json; estilo único de tabelas/cores
│   ├── io.py                 # leitura/escrita canônica de previsões e artefatos
│   ├── eda.py                # descritivas, estacionariedade (ADF/KPSS), deflação IPCA
│   ├── models.py             # ajustadores do núcleo + validação por origem móvel
│   ├── evaluation.py         # MASE/MAPE/MAE, tabela consolidada, heatmap
│   ├── benchmarks.py         # comparação com a prefeitura e com Oliveira (2024)
│   ├── generalization.py     # conjunto ampliado (31 séries)
│   ├── model_reports.py      # tabelas de parâmetros (ETS/SARIMA)
│   └── plotting.py           # estilo matplotlib do documento
├── scripts/
│   ├── run_pipeline_full.py         # pipeline CANÔNICO de 6 modelos (por estágio/venv)
│   ├── build_tex_artifacts.py       # regenera TODAS as tabelas/figuras do cache
│   ├── fig_hierarquia_temporal.py   # figura da Seção 5.5 (mesma fonte do checker)
│   ├── check_winrates.py            # assevera os 16 números canônicos da prosa
│   ├── report_variantes.py          # placar da bateria de variantes (Cap. 6)
│   ├── check_impute_sensitivity.py  # sensibilidade da imputação (Camaçari-ISSQN)
│   ├── run_pipeline.py              # núcleo histórico de 4 modelos (--force p/ cache)
│   └── verificacao_independente.py  # recomputa os resultados centrais sem o pacote
├── notebooks/01..04.ipynb    # exploração original (a via canônica são os scripts)
├── bahia/                    # extensão estadual (230 municípios; ver bahia/README.md)
├── data/forecasts/cv_all.csv # cache canônico da validação por origem móvel
├── data/forecasts/experiments/  # caches das variantes experimentais (Cap. 6)
├── data/reports/             # artefatos de proveniência (hierarquia, variantes, imputação)
└── tests/                    # 16 testes (não-vazamento, MASE, deflação, agregação)
```

## Uso

```bash
# Instalar (editable, com o coletor irmão)
pip install -e ../siconfi-collector
pip install -e .

# Regenerar todas as tabelas e figuras a partir do cache (~2 min)
python scripts/build_tex_artifacts.py --all
```

O passo `--models` re-fita ETS/Theta via `statsforecast` (extra opcional):
instale `pip install -r requirements-sf-lock.txt` num venv dedicado se o env
principal não o tiver. Os demais passos leem apenas o cache.

Artefatos vão para `../tcc-latex/tables/generated/` e
`../tcc-latex/figures/generated/`, conforme o `.tcc-pipeline.json` da raiz.

## Estado atual (2026-07-10)

Pipeline completo, sem stubs; `pytest` com 16 testes verdes (não-vazamento da
origem móvel, escala in-sample do MASE, identidade da deflação, imputação da
anomalia, agregação mensal→anual, smoke dos ajustadores). Os resultados
centrais são recomputáveis do zero por
`scripts/verificacao_independente.py`, que não importa o pacote.

Principais achados (Caps. 5–7 do TCC, números canônicos com previsão P1):

- Nenhum modelo domina. Em h = 1, ETS e Theta têm o melhor MASE mediano (0,62);
  em h = 12, **o Naïve sazonal é o melhor** (0,87) e nenhum modelo o supera
  (Ensemble 0,91, diferença dentro da banda de equivalência de 10%).
- No erro do total anual, o Ensemble supera a previsão da própria prefeitura em
  **80% dos anos-série do núcleo** (24/30; melhor modelo ex-post: 73%, 22/30),
  em 63% dos 155 anos-série do conjunto ampliado e em 67% dos 230 municípios
  da Bahia.
- O IPTU é muito mais previsível que o ISSQN; a vantagem sobre a projeção
  oficial concentra-se onde ela é frágil (municípios menores).

Veja `RUN_ORDER.md` (raiz do repo) para a sequência operacional completa.
