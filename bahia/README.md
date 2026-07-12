# Análise estadual: todos os municípios da Bahia

Análise independente do TCC (nenhum artefato do texto é alterado): aplica o
método do trabalho, coleta SICONFI, deflação IPCA, controle de qualidade e o
portfólio canônico de previsores, a **todos os municípios baianos**, e mede,
município a município, se a via mensal-agregada erra menos que a "Previsão
Atualizada" registrada pela própria prefeitura no RREO-Anexo 03.

## Pipeline

1. **Coleta** (uma vez, horas; retomável):

   ```bash
   cd siconfi-collector
   python -m siconfi collect rreo --state BA --years 2015-2025 \
       --annex "RREO-Anexo 03" --periods 1,2,6 --delay 0.6
   python -m siconfi transform-monthly
   python -m siconfi transform-prefeitura-forecast
   ```

   O período 6 carrega os doze meses do exercício (colunas `<MR-n>`); os
   períodos 1 e 2 preservam a Previsão Atualizada mais próxima da LOA
   (`prefeitura_forecast` usa o menor período disponível por ano).

2. **Benchmark** (`benchmark.py`): controle de qualidade idêntico ao da
   Seção 5.7 do TCC (cobertura ≥ 120 meses, valores positivos, no máximo dois
   anos anômalos imputados, lacunas de até dois meses interpoladas) e, nas
   séries aprovadas, o portfólio completo (Naïve sazonal, AutoETS, AutoARIMA
   sobre log, AutoTheta, Prophet e Ensemble pela média dos quatro formais)
   reestimado a cada origem de dezembro, prevendo os doze meses do exercício
   seguinte. Paraleliza por série (`--workers`).

3. **Figuras e números** (`figures.py`): confronto anual contra a projeção
   oficial (mesma regra do TCC), mapa coroplético da vantagem por município,
   vantagem contra população e taxa de vitória por faixa de porte.
   `geodata.py` baixa e cacheia a malha municipal do IBGE (sem geopandas) e lê
   a população do registro de entes do coletor.

4. **Ensemble fixo *vs.* melhor modelo** (`best_model.py`): compara o *Ensemble*
   fixo com a escolha do melhor modelo por série, tanto *ex-post* (estilo
   Oliveira, com hindsight) quanto *walk-forward* (só com o passado, operável).
   Mostra que o ganho aparente da seleção por série vem do uso do futuro.

5. **Relatório** (`report.py`): monta o `RELATORIO.md` a partir de
   `resumo.json`, `qc_log.txt` e `best_model.json`, com os números sempre
   sincronizados aos dados (nunca digitados à mão).

## Saídas (`out/`)

| arquivo | conteúdo |
|---|---|
| `cv_bahia.csv` | previsão e realizado anuais por série, modelo e ano-alvo |
| `qc_log.txt` | relatório do controle de qualidade e falhas de ajuste |
| `fig_mapa_bahia.*` | mapa: vantagem do Ensemble em p.p. de erro anual |
| `fig_vantagem_populacao.*` | vantagem por município contra população (log) |
| `fig_winrate_faixas.*` | margem mediana e taxa de vitória por faixa de porte |
| `fig_melhor_modelo.*` | Ensemble fixo vs. melhor modelo (ex-post e walk-forward) |
| `resumo.json`, `best_model.json` | números citados no relatório |

O relatório interpretativo fica em [RELATORIO.md](RELATORIO.md).
