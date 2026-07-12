# Sensibilidade da imputacao: ISSQN de Camacari sem imputar 2016

> Gerado por `scripts/check_impute_sensitivity.py`. Portfolio completo
> re-executado sobre a serie SEM a imputacao de 2016, mesmo protocolo.
> Sustenta a frase do Cap. 5: a imputacao nao muda o vencedor nem a
> ordenacao.

| Modelo | MASE h=1 com/sem | MASE h=12 com/sem | Erro anual medio com/sem (%) |
|---|---|---|---|
| Naive | 1.29 / 0.50 | 1.15 / 0.45 | 9.8 / 9.8 |
| ETS | 0.62 / 0.32 | 1.41 / 0.48 | 7.3 / 5.5 |
| SARIMA | 0.91 / 0.35 | 1.32 / 0.65 | 7.7 / 9.1 |
| Prophet | 0.99 / 0.37 | 1.83 / 0.46 | 8.1 / 7.6 |
| Theta | 0.65 / 0.36 | 1.39 / 0.44 | 7.5 / 6.5 |
| Ensemble | 0.71 / 0.28 | 1.36 / 0.53 | 7.6 / 4.4 |

Ordenacao por mase_h1: com imputacao ETS > Theta > Ensemble > SARIMA > Prophet > Naive; sem imputacao Ensemble > ETS > SARIMA > Theta > Prophet > Naive.

Ordenacao por mase_h12: com imputacao Naive > SARIMA > Ensemble > Theta > ETS > Prophet; sem imputacao Theta > Naive > Prophet > ETS > Ensemble > SARIMA.

Ordenacao por err_anual: com imputacao ETS > Theta > Ensemble > SARIMA > Prophet > Naive; sem imputacao Ensemble > ETS > Theta > Prophet > SARIMA > Naive.

ATENCAO: sem a imputacao o vencedor muda ALEM da banda de equivalencia de 10% em: mase_h1: ETS (0.32) perde para Ensemble (0.28) por mais de 10%; err_anual: ETS (5.54) perde para Ensemble (4.42) por mais de 10%. A frase do Cap. 5 nao se sustenta como esta.
