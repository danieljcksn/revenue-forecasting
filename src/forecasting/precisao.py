"""Fitters do portfolio canonico de seis modelos (consolidacao de _precisao_run).

Este modulo torna reproduzivel, a partir do pacote, o portfolio reportado no
TCC: Naive sazonal, AutoETS 'ZZZ', SARIMA D=1 (statsmodels, ordens congeladas
na janela inicial), AutoTheta, Prophet mensal corrigido e Ensemble (media
simples dos quatro formais). Historico e justificativas das escolhas: ver o
Cap. 4 do TCC e docs/decisoes em data/reports/.

Dois ambientes sao necessarios (numba/statsforecast nao suporta o Python do
venv principal): AutoETS/AutoTheta rodam no venv "precisao"
(requirements-sf-lock.txt); Naive/SARIMA/Prophet no venv principal
(requirements-lock.txt). O orquestrador scripts/run_pipeline_full.py divide o
trabalho em estagios por ambiente. Imports pesados sao preguicosos: importar
este modulo nao exige nenhuma das bibliotecas de modelagem.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.evaluation import seasonal_naive_insample_mae
from forecasting.models import (
    INITIAL_WINDOW,
    SEASON,
    FittedModel,
    _forecast_index,
    fit_sarimax_fixed,
)

for _noisy in ("prophet", "cmdstanpy"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ---------------- statsforecast (venv "precisao") -------------------------


@dataclass
class SFSpec:
    name: str
    make: object          # () -> instancia de modelo statsforecast
    use_log: bool = False


def make_sf_specs() -> dict[str, SFSpec]:
    """Especificacoes statsforecast ADOTADAS no portfolio (ETS e Theta).

    Import preguicoso: so funciona no venv com statsforecast instalado.
    """
    from statsforecast.models import AutoETS, AutoTheta
    return {
        "ETS": SFSpec("ETS", lambda: AutoETS(season_length=SEASON, model="ZZZ"),
                      use_log=False),
        "Theta": SFSpec("Theta", lambda: AutoTheta(season_length=SEASON),
                        use_log=False),
    }


def sf_rolling_cv(series: pd.Series, spec: SFSpec,
                  initial_window: int = INITIAL_WINDOW,
                  max_horizon: int = SEASON, step: int = 1,
                  season: int = SEASON) -> pd.DataFrame:
    """Copia fiel de rolling_origin_cv guiando um modelo statsforecast.

    Mesmo protocolo e mesmo denominador do MASE (escala in-sample do Naive
    sazonal por origem).
    """
    y = pd.Series(np.asarray(series, dtype=float),
                  index=pd.DatetimeIndex(series.index))
    n = len(y)
    rows = []
    for o in range(initial_window, n, step):
        train = y.iloc[:o]
        steps = min(max_horizon, n - o)
        if steps < 1:
            break
        scale = seasonal_naive_insample_mae(train, season=season)
        yv = np.asarray(train, dtype=float)
        base = np.log(yv) if spec.use_log else yv
        model = spec.make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.fit(base)
            pred = model.predict(h=steps)["mean"]
        pred = np.asarray(pred, dtype=float)
        if spec.use_log:
            pred = np.exp(pred)
        train_end = train.index[-1]
        for k in range(steps):
            rows.append({"origin": train_end, "train_end": train_end,
                         "step": k + 1, "target_date": y.index[o + k],
                         "y_true": float(y.iloc[o + k]),
                         "y_pred": float(pred[k]), "insample_scale": scale})
    return pd.DataFrame(rows)


# ---------------- Prophet mensal corrigido (venv principal) ----------------


def make_prophet_fitter(use_holidays: bool = False, yearly_fourier: int = 6,
                        cps: float = 0.05, cr: float = 0.8,
                        seasonality_mode: str = "multiplicative"):
    """Prophet na especificacao mensal ADOTADA: sem feriados de resolucao
    diaria, Fourier anual 6 (limite de Nyquist para 12 obs/ciclo) e modo
    multiplicativo. Escolhas de desenho documentadas no Cap. 4."""
    from prophet import Prophet

    def _fit(train, season: int = SEASON) -> FittedModel:
        t0 = time.perf_counter()
        idx = pd.DatetimeIndex(train.index)
        dfp = pd.DataFrame({"ds": idx, "y": np.asarray(train, dtype=float)})
        m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=yearly_fourier,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=cps, changepoint_range=cr)
        if use_holidays:
            m.add_country_holidays(country_name="BR")
        m.fit(dfp)
        return FittedModel(name="Prophet", fit_object=m,
                           params={"kind": "prophet",
                                   "last_train_date": pd.Timestamp(idx[-1])},
                           aic=None, train_seconds=time.perf_counter() - t0)
    return _fit


# ---------------- SARIMA D=1 (venv principal) ------------------------------


def make_sarima_D1_fitter(initial_train, season: int = SEASON,
                          use_log: bool = True, D: int = 1):
    """Seleciona as ordens com D FORCADO (auto_arima, AICc/KPSS) na janela
    inicial e as congela; a cada origem reestimam-se apenas os coeficientes.
    O D=1 e escolha de desenho do estudo (F_S alto na EDA; o OCSB
    subdiagnosticava a diferenciacao sazonal)."""
    import pmdarima as pm
    base = (np.log(np.asarray(initial_train, dtype=float)) if use_log
            else np.asarray(initial_train, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(base, seasonal=True, m=season, D=D,
                              test="kpss", stepwise=True,
                              suppress_warnings=True, error_action="ignore",
                              max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2,
                              n_jobs=1, information_criterion="aicc")
    order, sorder = tuple(model.order), tuple(model.seasonal_order)

    @dataclass
    class _F:
        order: tuple
        seasonal_order: tuple
        use_log: bool = True

        def __call__(self, train):
            return fit_sarimax_fixed(train, self.order, self.seasonal_order,
                                     use_log=self.use_log)

    f = _F(order, sorder, use_log)
    f.selected = (order, sorder)
    return f


def sarima_D1_rolling_with_var(series: pd.Series,
                               initial_window: int = INITIAL_WINDOW,
                               max_horizon: int = SEASON,
                               season: int = SEASON) -> pd.DataFrame:
    """Trilho SARIMA D=1 completo, capturando mu e sigma em log por passo.

    Retorna, por (origin, step): y_pred_median = exp(mu) (ponto mensal, otimo
    sob MAE/MASE, e o que entra no cv_all) e y_pred_mean =
    exp(mu + sigma^2/2) (correcao log-normal de Jensen, usada SO no agregado
    anual). Uma unica passada alimenta cv_all e sarima_var.
    """
    fitter = make_sarima_D1_fitter(series.iloc[:initial_window],
                                   season=season)
    order, sorder = fitter.selected
    y = pd.Series(np.asarray(series, dtype=float),
                  index=pd.DatetimeIndex(series.index))
    n = len(y)
    rows = []
    for o in range(initial_window, n, 1):
        train = y.iloc[:o]
        steps = min(max_horizon, n - o)
        if steps < 1:
            break
        scale = seasonal_naive_insample_mae(train, season=season)
        fm = fit_sarimax_fixed(train, order, sorder, use_log=True)
        fc = fm.fit_object.get_forecast(steps)
        mu = np.asarray(fc.predicted_mean, dtype=float)
        sig = np.asarray(fc.se_mean, dtype=float)
        train_end = train.index[-1]
        for k in range(steps):
            rows.append({"origin": train_end, "train_end": train_end,
                         "step": k + 1, "target_date": y.index[o + k],
                         "y_true": float(y.iloc[o + k]),
                         "mu_log": mu[k], "sigma_log": sig[k],
                         "y_pred_median": float(np.exp(mu[k])),
                         "y_pred_mean": float(np.exp(mu[k] + 0.5 * sig[k] ** 2)),
                         "insample_scale": scale})
    out = pd.DataFrame(rows)
    out.attrs["selected_orders"] = (order, sorder)
    return out


# ---------------- hierarquia temporal (Secao 5.5) --------------------------
# Fonte unica do experimento mensal-agregada vs anual-direta: usada pela
# figura fig_hierarquia_temporal e pelo scripts/check_winrates.py.

HIER_FAM_MENSAL = {"SARIMA": "ARIMA", "ETS": "ETS", "Naive": "Naive/RW"}
HIER_FAM_ANUAL = {"rw": "Naive/RW", "holt": "ETS", "holt_damp": "ETS",
                  "arima": "ARIMA"}
HIER_TEST_YEARS = (2021, 2022, 2023, 2024, 2025)


def hierarquia_mensal(cfg) -> pd.DataFrame:
    """Erros anuais da via mensal-agregada por familia (ARIMA/ETS/Naive).

    Le o cache canonico; o agregado do SARIMA usa a correcao de Jensen
    (soma de exp(mu+sigma^2/2) de data/forecasts/sarima_var.csv). Retorna
    colunas: serie, modelo, fam, ty, err.
    """
    cv = pd.read_csv(cfg.forecasts_dir / "cv_all.csv", parse_dates=["origin"])
    cv["serie"] = cv["municipio"] + "-" + cv["tributo"]
    dec = cv[(cv["origin"].dt.month == 12) & (cv["step"].between(1, 12))].copy()
    dec["ty"] = dec["origin"].dt.year + 1
    g = (dec.groupby(["serie", "modelo", "ty"])
         .agg(pred=("y_pred", "sum"), real=("y_true", "sum"),
              n=("step", "count")).reset_index())
    g = g[g["n"] == 12]
    var = pd.read_csv(cfg.forecasts_dir / "sarima_var.csv",
                      parse_dates=["origin"])
    var["serie"] = var["municipio"] + "-" + var["tributo"]
    vd = var[(var["origin"].dt.month == 12) & (var["step"].between(1, 12))].copy()
    vd["ty"] = vd["origin"].dt.year + 1
    vg = (vd.groupby(["serie", "ty"])
          .agg(pred=("y_pred_mean", "sum"), real=("y_true", "sum"),
               n=("step", "count")).reset_index())
    vg = vg[vg["n"] == 12]
    vg["modelo"] = "SARIMA"
    g = pd.concat([g[g["modelo"] != "SARIMA"], vg], ignore_index=True)
    g["err"] = 100 * (g["pred"] - g["real"]).abs() / g["real"].abs()
    g["fam"] = g["modelo"].map(HIER_FAM_MENSAL)
    return g[g["fam"].notna()].copy()


def hierarquia_anual(cfg) -> pd.DataFrame:
    """Erros anuais da via anual-direta: rw/holt/holt amortecido/ARIMA(log)
    re-ajustados sobre os totais anuais de cada serie (2021 a 2025, janela
    minima de 6 anos, sem drift). Retorna: serie, fam, err."""
    from forecasting.eda import prepare_series

    def _fit_pred(y: np.ndarray, model: str) -> float:
        if model == "rw":
            return float(y[-1])
        if model in ("holt", "holt_damp"):
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = ExponentialSmoothing(
                        y, trend="add", damped_trend=(model == "holt_damp"),
                        seasonal=None, initialization_method="estimated").fit()
                return float(r.forecast(1)[0])
            except Exception:
                return float(y[-1])
        if model == "arima":
            import pmdarima as pm
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdl = pm.auto_arima(np.log(y), seasonal=False,
                                        stepwise=True, suppress_warnings=True,
                                        error_action="ignore", max_p=2,
                                        max_q=2, max_d=2,
                                        information_criterion="aicc")
                return float(np.exp(mdl.predict(1)[0]))
            except Exception:
                return float(y[-1])
        raise ValueError(model)

    series = prepare_series(cfg, impute=True)
    rows = []
    for (mk, tr), s in series.items():
        tot = s.groupby(s.index.year).sum()
        tot = tot[tot.index <= HIER_TEST_YEARS[-1]]
        for year in HIER_TEST_YEARS:
            if year not in tot.index:
                continue
            hist = tot[tot.index < year]
            if len(hist) < 6:
                continue
            real = float(tot.loc[year])
            for mdl, fam in HIER_FAM_ANUAL.items():
                p = _fit_pred(hist.to_numpy(dtype=float), mdl)
                rows.append({"serie": f"{mk}-{tr}", "fam": fam,
                             "err": 100 * abs(p - real) / abs(real)})
    return pd.DataFrame(rows)


# ---------------- runner generico (kinds novos) ----------------------------


def forecast_any(model: FittedModel, horizon: int) -> pd.Series:
    """Despacho de previsao que cobre os kinds do pacote e os deste modulo."""
    from forecasting.models import forecast as base_forecast
    kind = model.params["kind"]
    if kind in ("naive", "ets", "sarimax", "prophet"):
        return base_forecast(model, horizon)
    index = _forecast_index(model.params["last_train_date"], horizon)
    if kind == "theta":
        vals = np.asarray(model.fit_object.forecast(horizon), dtype=float)
    else:
        raise ValueError(kind)
    return pd.Series(vals, index=index, name=model.name)


def rolling_cv_any(series: pd.Series, fit_fn,
                   initial_window: int = INITIAL_WINDOW,
                   max_horizon: int = SEASON, step: int = 1,
                   season: int = SEASON) -> pd.DataFrame:
    """Copia fiel de rolling_origin_cv chamando forecast_any."""
    y = pd.Series(np.asarray(series, dtype=float),
                  index=pd.DatetimeIndex(series.index))
    n = len(y)
    rows = []
    for o in range(initial_window, n, step):
        train = y.iloc[:o]
        steps = min(max_horizon, n - o)
        if steps < 1:
            break
        scale = seasonal_naive_insample_mae(train, season=season)
        fitted = fit_fn(train)
        preds = forecast_any(fitted, steps)
        train_end = train.index[-1]
        for k in range(steps):
            rows.append({"origin": train_end, "train_end": train_end,
                         "step": k + 1, "target_date": y.index[o + k],
                         "y_true": float(y.iloc[o + k]),
                         "y_pred": float(preds.iloc[k]),
                         "insample_scale": scale})
    return pd.DataFrame(rows)
