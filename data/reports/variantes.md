# Bateria de variantes: proveniencia do paragrafo do Cap. 6

> Gerado por `scripts/report_variantes.py` a partir de
> `data/forecasts/experiments/cv_exp_*.csv` (variantes re-ajustadas sob
> o mesmo protocolo) e do cache canonico. Sufixos: w72 = janela
> DESLIZANTE de 72 meses (reage mais depressa ao degrau de nivel);
> log = ajuste sobre o logaritmo; damped = tendencia amortecida.

## A frase do texto

A variante da frase e a **Ens5 w72+Naive**: MASE mediano h=12 de
0.91 (Ensemble canonico) para 0.84, e
vitorias anuais de 24/30 para 19/30.
Reagir ao degrau melhora o decimo segundo mes e custa o total anual.

## Placar completo

| Variante | MASE h=1 | MASE h=12 | Vitorias anuais | Erro anual medio (%) |
|---|---|---|---|---|
| Naive | 0.888 | 0.873 | 19/30 | 10.8 |
| ETS | 0.619 | 0.965 | 21/30 | 11.1 |
| SARIMA | 0.748 | 0.971 | 16/30 | 14.3 |
| Theta | 0.619 | 0.951 | 19/30 | 11.9 |
| Prophet | 0.749 | 0.962 | 22/30 | 11.2 |
| Ensemble | 0.681 | 0.905 | 24/30 | 10.9 |
| NaiveDrift | 0.843 | 0.839 | 18/30 | 14.3 |
| ETS_damped | 0.625 | 0.957 | 22/30 | 13.4 |
| ETS_log | 0.640 | 0.945 | 20/30 | 12.9 |
| Theta_log | 0.634 | 0.850 | 17/30 | 14.0 |
| ETS_w72 | 0.633 | 0.926 | 18/30 | 11.1 |
| SARIMA_w72 | 0.709 | 1.100 | 17/30 | 15.8 |
| Theta_w72 | 0.656 | 0.957 | 16/30 | 12.8 |
| Ens4 canonico | 0.681 | 0.905 | 24/30 | 10.9 |
| Ens5 +Naive | 0.684 | 0.889 | 24/30 | 10.5 |
| Ens5 +NaiveDrift | 0.669 | 0.882 | 23/30 | 10.1 |
| Ens4 w72 | 0.640 | 0.879 | 18/30 | 11.6 |
| Ens5 w72+Naive | 0.676 | 0.844 | 19/30 | 11.3 |
| Ens4 ETS_log | 0.686 | 0.870 | 22/30 | 11.3 |
| Ens4 logs | 0.660 | 0.871 | 19/30 | 12.2 |
| Ens5 logs+Naive | 0.663 | 0.866 | 19/30 | 11.8 |
| Ens3 ETS+Theta+Naive | 0.663 | 0.928 | 20/30 | 10.5 |
| Ens2 ETS+Naive | 0.726 | 0.984 | 21/30 | 10.3 |
| Ens5 +NaiveDrift (mediana) | 0.592 | 0.899 | 22/30 | 10.3 |
| Ens5 +Naive (mediana) | 0.612 | 0.895 | 24/30 | 10.7 |
