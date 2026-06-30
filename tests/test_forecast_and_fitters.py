"""Smoke tests for model forecasting and fitter construction.

The heavier model backends are imported lazily so the suite remains usable in a
minimal development environment while still exercising ETS, SARIMA, and Prophet
when their dependencies are installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import models as M


def _short_series(n: int = 48, start: str = "2018-01-01") -> pd.Series:
    """Return a positive monthly series with trend and seasonality."""
    idx = pd.date_range(start, periods=n, freq="MS")
    t = np.arange(n, dtype=float)
    y = 1000.0 + 4.0 * t + 30.0 * np.sin(2.0 * np.pi * (t % 12) / 12)  # > 0
    return pd.Series(y, index=idx, name="y")


def _smoke(fit_fn, kind: str, horizon: int = 12) -> None:
    """Fit a model, forecast ahead, and validate output shape and finiteness."""
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
    _smoke(M.fit_sarima, "sarimax")


def test_forecast_prophet_smoke():
    pytest.importorskip("prophet")
    _smoke(M.fit_prophet, "prophet")


def test_make_fitters_returns_four(monkeypatch):
    """The fitter factory exposes the canonical four-model core."""
    monkeypatch.setattr(M, "make_sarima_fitter", lambda s: ("SARIMA_FIT", len(s)))
    fitters = M.make_fitters(_short_series(80))

    assert list(fitters) == ["Naive", "ETS", "Prophet", "SARIMA"]
    assert fitters["Naive"] is M.fit_naive_seasonal
    assert fitters["ETS"] is M.fit_ets
    assert fitters["Prophet"] is M.fit_prophet
    assert fitters["SARIMA"] == ("SARIMA_FIT", M.INITIAL_WINDOW)
