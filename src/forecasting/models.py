"""Model fitting and forecasting wrappers.

Cada modelo expoe interface uniforme:
    fit(train_series) -> FittedModel
    forecast(fitted_model, horizon) -> pd.Series

Core models covered by this module:
- Baseline:             Naive Sazonal
- Estatistico classico: ETS (Holt-Winters), SARIMA (auto_arima)
- Estrutural:           Prophet

Bibliotecas:
- statsmodels (ETS, SARIMAX)
- pmdarima (auto_arima)
- prophet

The canonical forecast cache may contain additional predictors, such as Theta
and Ensemble. This module keeps the portable four-model implementation used by
the local training driver.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from forecasting.config import PipelineConfig

SEASON = 12  # Annual seasonality of monthly series.

# Rolling-origin validation parameters for an expanding training window.
# INITIAL_WINDOW = 72 keeps six full seasonal cycles in the first training set.
# MAX_HORIZON = 12 stores a full one-year forecast path from each origin.
# ROLLING_STEP = 1 advances the origin by one month.
INITIAL_WINDOW = 72
MAX_HORIZON = SEASON
ROLLING_STEP = 1


def _forecast_index(last_train_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Datas (inicio de mes) dos `horizon` meses seguintes ao fim do treino."""
    start = pd.Timestamp(last_train_date) + pd.offsets.MonthBegin(1)
    return pd.date_range(start=start, periods=horizon, freq="MS")


# ---------- Tipos auxiliares ---------------------------------------------


@dataclass
class FittedModel:
    """Trained model plus lightweight reporting metadata."""
    name: str
    fit_object: Any
    params: dict[str, Any]
    aic: float | None
    train_seconds: float


# ---------- Reprodutibilidade --------------------------------------------


def set_global_seeds(seed: int = 42) -> None:
    """Seta seeds em numpy e random (numpy ja importado no topo do modulo)."""
    import random
    random.seed(seed)
    np.random.seed(seed)


# ---------- Wrappers por modelo ------------------------------------------


def fit_naive_seasonal(train: pd.Series, season: int = SEASON) -> FittedModel:
    """Seasonal Naive: y_{T+h} = y_{T+h-s}.

    Each forecast repeats the value from the same month in the last observed
    seasonal cycle. There are no fitted parameters.
    """
    t0 = time.perf_counter()
    y = np.asarray(train, dtype=float)
    last_cycle = y[-season:]
    return FittedModel(
        name="Naive",
        fit_object=last_cycle,
        params={"kind": "naive", "season": season,
                "last_train_date": pd.Timestamp(train.index[-1])},
        aic=None,
        train_seconds=time.perf_counter() - t0,
    )


# ETS specifications evaluated by AICc at each origin. The grid keeps additive
# error and varies trend, dampening, and seasonal form.
_ETS_GRID = [
    dict(error="add", trend="add", damped_trend=False, seasonal="add"),
    dict(error="add", trend="add", damped_trend=True, seasonal="add"),
    dict(error="add", trend=None, damped_trend=False, seasonal="add"),
    dict(error="add", trend="add", damped_trend=False, seasonal="mul"),
    dict(error="add", trend="add", damped_trend=True, seasonal="mul"),
    dict(error="add", trend=None, damped_trend=False, seasonal="mul"),
]


def fit_ets(train: pd.Series, season: int = SEASON) -> FittedModel:
    """ETS (exponential smoothing / Holt-Winters) via statsmodels.

    Selects the lowest-AICc specification from a small, explicit grid at each
    rolling origin.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    t0 = time.perf_counter()
    y = pd.Series(np.asarray(train, dtype=float),
                  index=pd.DatetimeIndex(train.index)).asfreq("MS")
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in _ETS_GRID:
            try:
                res = ETSModel(y, seasonal_periods=season, **spec).fit(disp=False)
            except Exception:
                continue
            aicc = getattr(res, "aicc", None)
            if aicc is None or not np.isfinite(aicc):
                continue
            if best is None or aicc < best[0]:
                best = (float(aicc), res, spec)
    if best is None:
        raise RuntimeError("nenhuma especificacao ETS convergiu para esta serie")
    aicc, res, spec = best
    # ETS(E,T,S) taxonomy: additive error; optional trend/damping; additive or
    # multiplicative seasonality.
    e = "A"
    t = "Ad" if spec["damped_trend"] else ("A" if spec["trend"] else "N")
    sea = "A" if spec["seasonal"] == "add" else "M"
    label = f"ETS({e},{t},{sea})"
    return FittedModel(
        name="ETS",
        fit_object=res,
        params={"kind": "ets", "spec": label, "aicc": round(aicc, 1),
                "last_train_date": pd.Timestamp(train.index[-1]), **spec},
        aic=float(getattr(res, "aic", float("nan"))),
        train_seconds=time.perf_counter() - t0,
    )


def select_sarima_order(train: pd.Series, season: int = SEASON):
    """Seleciona as ordens (p,d,q)(P,D,Q) via pmdarima.auto_arima (stepwise).

    Regular and seasonal differencing follow KPSS and OCSB tests; stepwise
    search minimizes AICc under bounded orders.
    """
    import pmdarima as pm

    y = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            y, seasonal=True, m=season,
            test="kpss", seasonal_test="ocsb",
            stepwise=True, suppress_warnings=True, error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2, max_D=1,
            n_jobs=1, information_criterion="aicc",
        )
    return tuple(model.order), tuple(model.seasonal_order)


def fit_sarima(train: pd.Series, season: int = SEASON, use_log: bool = True) -> FittedModel:
    """SARIMA com ordens selecionadas pelo auto_arima na propria serie.

    Used for a single full-sample fit after order selection, then delegates to
    ``fit_sarimax_fixed``.
    """
    base = np.log(np.asarray(train, dtype=float)) if use_log \
        else np.asarray(train, dtype=float)
    order, seasonal_order = select_sarima_order(
        pd.Series(base, index=train.index), season=season)
    return fit_sarimax_fixed(train, order, seasonal_order, use_log=use_log)


def order_to_seasonal(seasonal_order):
    """Normaliza a ordem sazonal para a tupla (P,D,Q,s) do statsmodels."""
    so = tuple(seasonal_order)
    return so if len(so) == 4 else (*so[:3], SEASON)


def fit_sarimax_fixed(train: pd.Series, order, seasonal_order,
                      use_log: bool = True) -> FittedModel:
    """Fit SARIMAX with fixed orders and coefficients estimated on the train set.

    When ``use_log=True``, fitting happens on log revenue and forecasts return
    to the original scale with ``exp``. The back-transformed point forecast is
    the log-normal median, which is appropriate for absolute-error metrics.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    t0 = time.perf_counter()
    raw = np.asarray(train, dtype=float)
    z = np.log(raw) if use_log else raw
    y = pd.Series(z, index=pd.DatetimeIndex(train.index)).asfreq("MS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, order=tuple(order),
                      seasonal_order=order_to_seasonal(seasonal_order),
                      enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
    return FittedModel(
        name="SARIMA",
        fit_object=res,
        params={"kind": "sarimax", "order": tuple(order),
                "seasonal_order": tuple(seasonal_order), "log_transform": use_log,
                "last_train_date": pd.Timestamp(train.index[-1])},
        aic=float(res.aic),
        train_seconds=time.perf_counter() - t0,
    )


@dataclass
class _SarimaFitter:
    """Callable SARIMA fitter with orders fixed after initial selection."""
    order: tuple
    seasonal_order: tuple
    use_log: bool = True

    def __call__(self, train: pd.Series) -> FittedModel:
        return fit_sarimax_fixed(train, self.order, self.seasonal_order,
                                 use_log=self.use_log)


def make_sarima_fitter(initial_train: pd.Series, season: int = SEASON,
                       use_log: bool = True) -> "FitFn":
    """Build the SARIMA ``fit_fn`` used by rolling-origin validation.

    Orders are selected once on the initial training window. The returned
    callable refits only SARIMAX coefficients at each later origin.
    """
    base = np.log(np.asarray(initial_train, dtype=float)) if use_log \
        else np.asarray(initial_train, dtype=float)
    order, seasonal_order = select_sarima_order(
        pd.Series(base, index=initial_train.index), season=season)
    return _SarimaFitter(order, seasonal_order, use_log)


def fit_prophet(train: pd.Series, season: int = SEASON) -> FittedModel:
    """Prophet additive-structure model with Brazilian holidays.

    Uses yearly seasonality for monthly data, disables weekly/daily components,
    includes Brazilian holidays, and keeps Prophet's default priors.
    """
    import logging

    from prophet import Prophet

    for noisy in ("prophet", "cmdstanpy"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    t0 = time.perf_counter()
    idx = pd.DatetimeIndex(train.index)
    dfp = pd.DataFrame({"ds": idx, "y": np.asarray(train, dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative")
        m.add_country_holidays(country_name="BR")
        m.fit(dfp)
    return FittedModel(
        name="Prophet",
        fit_object=m,
        params={"kind": "prophet", "seasonality_mode": "multiplicative",
                "holidays": "BR", "last_train_date": pd.Timestamp(idx[-1])},
        aic=None,
        train_seconds=time.perf_counter() - t0,
    )


# ---------- Fabricas/regimes compartilhados pelos drivers ----------------


def make_fitters(s: pd.Series) -> dict:
    """Build the core fitters for a series in canonical order.

    SARIMA orders are fixed on the first 72 observations and coefficients are
    refit at each origin.
    """
    return {
        "Naive": fit_naive_seasonal,
        "ETS": fit_ets,
        "Prophet": fit_prophet,
        "SARIMA": make_sarima_fitter(s.iloc[:INITIAL_WINDOW]),
    }


def covid_regime(target: pd.Timestamp, cfg: PipelineConfig) -> str:
    """Temporal regime for a target date, based on ``cfg.covid_period``:

      - ``"pre"``   : antes do inicio da pandemia (``< covid_period.start``)
      - ``"covid"`` : dentro da janela pandemica ``[start, end]``
      - ``"pos"``   : apos a janela (``> covid_period.end``)
    """
    start = pd.Timestamp(cfg.covid_period.start)
    end = pd.Timestamp(cfg.covid_period.end)
    if target < start:
        return "pre"
    if target <= end:
        return "covid"
    return "pos"


# ---------- Previsao -----------------------------------------------------


def forecast(model: FittedModel, horizon: int) -> pd.Series:
    """Generate a `horizon`-step forecast indexed by future month starts.

    Dispatch por ``model.params['kind']`` (um caminho por paradigma): ``naive``
    repete o ultimo ciclo sazonal; ``ets`` e ``sarimax`` usam ``.forecast`` do
    statsmodels (o SARIMAX volta do log por ``exp`` -- ver ``fit_sarimax_fixed``);
    ``prophet`` usa ``.predict``.
    """
    kind = model.params["kind"]
    index = _forecast_index(model.params["last_train_date"], horizon)

    match kind:
        case "naive":
            season = model.params["season"]
            last_cycle = np.asarray(model.fit_object, dtype=float)
            values = np.array([last_cycle[(k % season)] for k in range(horizon)])
        case "ets":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = np.asarray(model.fit_object.forecast(horizon), dtype=float)
        case "sarimax":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = np.asarray(model.fit_object.forecast(horizon), dtype=float)
            if model.params.get("log_transform"):
                values = np.exp(values)
        case "prophet":
            future = pd.DataFrame({"ds": index})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = model.fit_object.predict(future)
            values = pred["yhat"].to_numpy(dtype=float)
        case _:
            raise ValueError(f"tipo de modelo desconhecido: {kind!r}")

    return pd.Series(values, index=index, name=model.name)


# ---------- Validacao por origem movel -----------------------------------

FitFn = Callable[[pd.Series], FittedModel]


def rolling_origin_cv(
    series: pd.Series,
    fit_fn: FitFn,
    initial_window: int = INITIAL_WINDOW,
    max_horizon: int = MAX_HORIZON,
    step: int = ROLLING_STEP,
    season: int = SEASON,
) -> pd.DataFrame:
    """Rolling-origin validation with an expanding training window.

    Para cada origem ``o`` (numero de meses no treino) de ``initial_window``
    ate ``len(series) - 1``, avancando de ``step`` em ``step``:

      1. ajusta o modelo em ``series[:o]``;
      2. projeta o caminho de ate ``max_horizon`` passos a frente;
      3. registra, para cada passo com valor realizado disponivel, o par
         (previsto, realizado) e a escala in-sample do Naive Sazonal daquele
         treino (denominador do MASE).

    The full forecast path is kept for each origin. Metrics by horizon are
    obtained by filtering ``step == h``; annual benchmarks sum the twelve steps
    from December origins.

    Retorna um DataFrame longo com colunas: origin, train_end, step,
    target_date, y_true, y_pred, insample_scale.
    """
    from forecasting.evaluation import seasonal_naive_insample_mae

    y = pd.Series(np.asarray(series, dtype=float),
                  index=pd.DatetimeIndex(series.index))
    n = len(y)
    rows: list[dict[str, Any]] = []

    for o in range(initial_window, n, step):
        train = y.iloc[:o]
        steps_avail = min(max_horizon, n - o)
        if steps_avail < 1:
            break
        scale = seasonal_naive_insample_mae(train, season=season)
        fitted = fit_fn(train)
        preds = forecast(fitted, steps_avail)
        train_end = train.index[-1]
        for k in range(steps_avail):
            target_date = y.index[o + k]
            rows.append({
                "origin": train_end,
                "train_end": train_end,
                "step": k + 1,
                "target_date": target_date,
                "y_true": float(y.iloc[o + k]),
                "y_pred": float(preds.iloc[k]),
                "insample_scale": scale,
            })
    return pd.DataFrame(rows)


def run_all(cfg: PipelineConfig) -> list[Path]:
    """Generate model-parameter tables and diagnostic figures.

    Reads the rolling-origin cache and performs full-series fits only for
    parameter and diagnostic reporting.
    """
    from forecasting import model_reports
    return model_reports.run_all(cfg)
