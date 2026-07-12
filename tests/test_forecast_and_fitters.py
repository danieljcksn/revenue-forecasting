"""Testes de fumaca dos modelos via ``forecast()`` e da fabrica ``make_fitters``.

A suite ``test_core.py`` so exercita o ramo Naive de ``forecast()``. Aqui
cobrem-se os tres ramos restantes (ETS, SARIMA/SARIMAX, Prophet) com um teste de
fumaca sintetico: confirma o ``kind``, o shape/horizonte e a ausencia de NaN/inf.
Os tres dependem de bibliotecas pesadas (statsmodels/pmdarima/prophet); por isso
usam ``pytest.importorskip`` e sao PULADOS automaticamente onde elas faltam --
mantendo a suite verde em qualquer maquina e exercitando-os onde houver deps.

``make_fitters`` e testado SEM dependencias pesadas (mock de ``make_sarima_fitter``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import models as M


def _short_series(n: int = 48, start: str = "2018-01-01") -> pd.Series:
    """Serie mensal curta, estritamente positiva, com tendencia e sazonalidade."""
    idx = pd.date_range(start, periods=n, freq="MS")
    t = np.arange(n, dtype=float)
    y = 1000.0 + 4.0 * t + 30.0 * np.sin(2.0 * np.pi * (t % 12) / 12)  # > 0
    return pd.Series(y, index=idx, name="y")


def _smoke(fit_fn, kind: str, horizon: int = 12) -> None:
    """Ajusta, preve ``horizon`` passos e confere kind/shape/finitude."""
    fm = fit_fn(_short_series())
    assert fm.params["kind"] == kind
    fc = M.forecast(fm, horizon)
    assert isinstance(fc, pd.Series)
    assert len(fc) == horizon
    assert fc.notna().all()
    assert np.isfinite(fc.to_numpy()).all()


def test_forecast_ets_smoke():
    pytest.importorskip("statsmodels")
    _smoke(M.fit_ets, "ets")


def test_forecast_sarima_smoke():
    pytest.importorskip("statsmodels")
    pytest.importorskip("pmdarima")
    _smoke(M.fit_sarima, "sarimax")  # fit_sarima -> fit_sarimax_fixed -> kind "sarimax"


def test_forecast_prophet_smoke():
    pytest.importorskip("prophet")
    _smoke(M.fit_prophet, "prophet")


def test_make_fitters_returns_four(monkeypatch):
    """``make_fitters`` devolve os QUATRO fitters na ordem canonica
    (Naive, ETS, Prophet, SARIMA); o SARIMA vem de ``make_sarima_fitter`` sobre a
    janela inicial ``INITIAL_WINDOW``. Mocka-se ``make_sarima_fitter`` para nao
    exigir pmdarima -- o objetivo e a estrutura da fabrica, nao o ajuste real.
    """
    monkeypatch.setattr(M, "make_sarima_fitter", lambda s: ("SARIMA_FIT", len(s)))
    fitters = M.make_fitters(_short_series(80))

    assert list(fitters) == ["Naive", "ETS", "Prophet", "SARIMA"]
    assert fitters["Naive"] is M.fit_naive_seasonal
    assert fitters["ETS"] is M.fit_ets
    assert fitters["Prophet"] is M.fit_prophet
    # SARIMA construido sobre s.iloc[:INITIAL_WINDOW] (72 observacoes).
    assert fitters["SARIMA"] == ("SARIMA_FIT", M.INITIAL_WINDOW)
