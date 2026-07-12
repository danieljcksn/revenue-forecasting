# Hierarquia temporal: proveniencia dos numeros da Secao 5.5

> Gerado por `scripts/check_winrates.py`. Recorte identico ao da
> figura fig_hierarquia_temporal: mensal = familias ARIMA (SARIMA com
> correcao de Jensen), ETS e Naive do cache canonico; anual-direta =
> rw/holt/holt amortecido/ARIMA(log) re-ajustados sobre os totais
> anuais, 2021 a 2025, sem drift.

## Painel A: erro anual medio por familia (%)

| Familia | Mensal-agregada | Anual-direta |
|---|---|---|
| ARIMA | 11.0 | 15.7 |
| ETS | 11.1 | 11.7 |
| Naive/RW | 10.8 | 10.8 |

## Painel B: melhor de cada via, por serie (%)

| Serie | Melhor mensal | Melhor anual |
|---|---|---|
| camacari-IPTU | 15.9 | 15.9 |
| camacari-ISSQN | 7.3 | 9.6 |
| ilheus-IPTU | 7.3 | 8.6 |
| ilheus-ISSQN | 13.5 | 13.5 |
| salvador-IPTU | 5.5 | 4.8 |
| salvador-ISSQN | 4.4 | 7.3 |

**Media do painel B: mensal 8.99% vs anual 9.97%**
(arredondadas para 9,0% e 10,0% na prosa da Secao 5.5).
