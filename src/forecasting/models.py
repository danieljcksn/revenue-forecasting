"""Wrappers de treinamento dos modelos do TCC.

Cada modelo expoe interface uniforme:
    fit(train_series) -> FittedModel
    forecast(fitted_model, horizon) -> pd.Series

Modelos cobertos (tres paradigmas, quatro especificacoes parcimoniosas):
- Baseline:             Naive Sazonal
- Estatistico classico: ETS (Holt-Winters), SARIMA (auto_arima)
- Estrutural:           Prophet

O paradigma de aprendizado profundo esta fora do escopo deste trabalho:
com 132 observacoes mensais por serie, modelos profundos tendem a
sobreajustar e exigem volumes de dados muito superiores. Fica registrado
apenas como direcao de extensao (ver Cap. 7 do TCC), sem implementacao
neste pacote.

Bibliotecas:
- statsmodels (ETS, SARIMAX)
- pmdarima (auto_arima)
- prophet

NOTA (reprodutibilidade): o portfolio reportado no TCC amplia este nucleo para
SEIS previsores -- acrescenta o metodo Theta e um Ensemble (media simples de
ETS/SARIMA/Theta/Prophet) -- e adota configuracoes corrigidas (AutoETS de
taxonomia completa, SARIMA com D=1 forcado, Prophet mensal sem feriados e
Fourier=6), produzidas via ``statsforecast`` pelos scripts de ``_precisao_run/``
e cacheadas em ``cv_all.csv``. Este modulo implementa o nucleo de quatro modelos.
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

SEASON = 12  # periodicidade sazonal anual das series mensais

# Parametros da validacao por origem movel (janela de treino expansiva).
# INITIAL_WINDOW = 72: exige 6 anos minimos de treino (6*12) antes da 1a origem,
#   o que da ~60 origens nas series de 132 meses sem treinar com pouca historia.
# MAX_HORIZON = 12: horizonte maximo avaliado, 1 ano (alinhado a LOA do exercicio).
# ROLLING_STEP = 1: a origem avanca um mes por vez (origem movel mensal).
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
    """Encapsula um modelo treinado + metadados para reportagem."""
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
    """Naive Sazonal: y_{T+h} = y_{T+h-s}.

    A previsao para cada mes futuro repete o valor do mesmo mes do ultimo
    ciclo observado. Nao ha parametros a estimar; serve de regua (denominador
    do MASE) e de piso que todo modelo "de verdade" precisa superar.
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


# Espaco de configuracoes ETS avaliadas por AICc a cada origem. Erro aditivo
# (estavel para series ja deflacionadas); tendencia aditiva com/sem
# amortecimento; sazonalidade aditiva ou multiplicativa (m=12). A selecao por
# AICc a cada dobra reproduz o procedimento de `forecast::ets` (Hyndman).
_ETS_GRID = [
    dict(error="add", trend="add", damped_trend=False, seasonal="add"),
    dict(error="add", trend="add", damped_trend=True, seasonal="add"),
    dict(error="add", trend=None, damped_trend=False, seasonal="add"),
    dict(error="add", trend="add", damped_trend=False, seasonal="mul"),
    dict(error="add", trend="add", damped_trend=True, seasonal="mul"),
    dict(error="add", trend=None, damped_trend=False, seasonal="mul"),
]


def fit_ets(train: pd.Series, season: int = SEASON) -> FittedModel:
    """ETS (suavizacao exponencial / Holt-Winters) via statsmodels.

    Seleciona, entre um conjunto de especificacoes plausiveis (tendencia
    amortecida ou nao; sazonalidade aditiva ou multiplicativa), aquela de
    menor AICc -- criterio que penaliza complexidade e e apropriado para
    amostras curtas. A re-selecao a cada dobra evita vazamento de informacao.
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
    # Taxonomia ETS(E,T,S): erro sempre aditivo (A); tendencia aditiva (A),
    # amortecida (Ad) ou ausente (N); sazonalidade aditiva (A) ou multiplicativa (M).
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

    A diferenciacao regular e sazonal segue os testes KPSS e OCSB; a busca
    stepwise minimiza o AICc (Hyndman & Khandakar, 2008). Limites de ordem
    modestos bastam para series mensais fiscais e mantem a busca barata.
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

    Usado para a tabela de parametros (ajuste unico na amostra completa), na
    mesma escala (log) da validacao por origem movel. Esta delega para
    ``fit_sarimax_fixed`` apos a selecao das ordens.
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
    """Ajusta um SARIMAX com ordens FIXAS (coeficientes reestimados no treino).

    Quando ``use_log=True``, o modelo e ajustado sobre o logaritmo da serie.
    Isso estabiliza a variancia das series de receita, cuja amplitude sazonal
    cresce com o nivel (sazonalidade multiplicativa): no logaritmo, essa
    estrutura torna-se aditiva e portanto compativel com o ARIMA, que e linear.
    A previsao e devolvida na escala original por ``exp``; como o exponencial da
    previsao no log corresponde a *mediana* da distribuicao log-normal (e nao a
    media), ele e exatamente o preditor pontual otimo sob erro absoluto
    (MAE/MASE) -- por isso NAO se aplica nenhuma correcao de vies de Jensen.
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
    """``fit_fn`` do SARIMA na origem movel: carrega as ordens (p,d,q)(P,D,Q)
    FIXADAS na janela inicial e, a cada origem, reestima apenas os coeficientes
    (instancia chamavel). Substitui o antigo atributo monkeypatch em ``_fit``."""
    order: tuple
    seasonal_order: tuple
    use_log: bool = True

    def __call__(self, train: pd.Series) -> FittedModel:
        return fit_sarimax_fixed(train, self.order, self.seasonal_order,
                                 use_log=self.use_log)


def make_sarima_fitter(initial_train: pd.Series, season: int = SEASON,
                       use_log: bool = True) -> "FitFn":
    """Fabrica o ``fit_fn`` do SARIMA para a validacao por origem movel.

    As ordens (p,d,q)(P,D,Q) sao selecionadas pelo ``auto_arima`` UMA UNICA VEZ,
    sobre a janela inicial de treino -- que so contem dados ate a primeira origem,
    portanto SEM VAZAMENTO de futuro. O fitter devolvido reestima, a cada origem,
    apenas os COEFICIENTES do SARIMAX (mantendo as ordens fixas). Decisao
    deliberada de custo x rigor: refazer a busca stepwise nas ~60 origens custaria
    ordens de grandeza a mais sem mudar materialmente as ordens. A serie e
    log-transformada (``use_log``) pela razao documentada em ``fit_sarimax_fixed``.
    """
    base = np.log(np.asarray(initial_train, dtype=float)) if use_log \
        else np.asarray(initial_train, dtype=float)
    order, seasonal_order = select_sarima_order(
        pd.Series(base, index=initial_train.index), season=season)
    return _SarimaFitter(order, seasonal_order, use_log)


def fit_prophet(train: pd.Series, season: int = SEASON) -> FittedModel:
    """Prophet (decomposicao estrutural) com feriados nacionais.

    Configuracao homogenea entre as series: apenas sazonalidade anual (as
    series sao mensais, logo sem componentes semanal/diario), feriados do
    Brasil via `add_country_holidays('BR')` e demais hiperparametros no default.
    A sazonalidade e MULTIPLICATIVA: a amplitude do ciclo das receitas cresce
    com o nivel da serie (sobretudo no IPTU), de modo que o modo aditivo
    subestimaria sistematicamente os picos -- mesma razao pela qual o SARIMA
    roda em log e o ETS admite componente sazonal multiplicativo. Sem tuning
    fino de priors (changepoints/escala no default): a evidencia das
    competicoes M indica que ele raramente compensa em series mensais curtas.
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
    """Fabrica UNICA dos quatro fitters do TCC para a serie ``s``, na ordem
    canonica (Naive, ETS, Prophet, SARIMA).

    O SARIMA tem as ordens fixadas na janela inicial de 72 observacoes (sem
    vazamento) e reestima apenas os coeficientes a cada origem. Usada tanto por
    ``scripts/run_pipeline.py`` quanto por ``generalization.run_generalization``
    -- antes cada driver reconstruia este mesmo dict a mao."""
    return {
        "Naive": fit_naive_seasonal,
        "ETS": fit_ets,
        "Prophet": fit_prophet,
        "SARIMA": make_sarima_fitter(s.iloc[:INITIAL_WINDOW]),
    }


def covid_regime(target: pd.Timestamp, cfg: PipelineConfig) -> str:
    """Regime temporal de uma data-alvo, segundo a janela COVID do cfg
    (``cfg.covid_period``). Definicao UNICA, usada tanto pela coluna ``regime``
    de ``cv_all.csv`` (run_pipeline) quanto pela nota de regime
    (``evaluation.covid_regime_note``):

      - ``"pre"``   : antes do inicio da pandemia (``< covid_period.start``)
      - ``"covid"`` : dentro da janela pandemica ``[start, end]``
      - ``"pos"``   : apos a janela (``> covid_period.end``)

    Como ``covid_period.end = 2021-12-31``, vale a equivalencia EXATA
    ``{pre, covid}`` == ano <= 2021 e ``{pos}`` == ano >= 2022 -- por isso a nota
    do Cap. 5 (cauda da pandemia ``<=2021`` vs normalizacao ``2022--2025``)
    deriva deste mesmo corte sem alterar nenhum numero."""
    start = pd.Timestamp(cfg.covid_period.start)
    end = pd.Timestamp(cfg.covid_period.end)
    if target < start:
        return "pre"
    if target <= end:
        return "covid"
    return "pos"


# ---------- Previsao -----------------------------------------------------


def forecast(model: FittedModel, horizon: int) -> pd.Series:
    """Gera previsao `horizon` passos a frente, indexada pelas datas futuras.

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
    """Validacao cruzada por origem movel com janela de treino expansiva.

    Para cada origem ``o`` (numero de meses no treino) de ``initial_window``
    ate ``len(series) - 1``, avancando de ``step`` em ``step``:

      1. ajusta o modelo em ``series[:o]``;
      2. projeta o caminho de ate ``max_horizon`` passos a frente;
      3. registra, para cada passo com valor realizado disponivel, o par
         (previsto, realizado) e a escala in-sample do Naive Sazonal daquele
         treino (denominador do MASE).

    Como o modelo e ajustado uma unica vez por origem e o caminho completo e
    guardado, a mesma execucao serve tanto as metricas por horizonte (filtrar
    ``step == h``) quanto a agregacao anual do benchmark da prefeitura (somar
    os 12 passos das origens que terminam em dezembro).

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
    """Gera as tabelas de parametros e figuras por modelo (Secao 5.2).

    Le o cache da validacao por origem movel produzido por
    ``scripts/run_pipeline.py`` (que faz o treino pesado uma unica vez) e
    ajusta cada modelo na serie completa para as tabelas de parametros e os
    diagnosticos. Nao re-executa a validacao por origem movel.
    """
    from forecasting import model_reports
    return model_reports.run_all(cfg)
