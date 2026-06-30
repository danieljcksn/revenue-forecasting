"""Core tests for metrics, deflation, anomaly treatment, and aggregation.

These tests protect invariants that would silently alter generated tables and
figures if broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_seasonal_naive_insample_mae_known_value():
    from forecasting.evaluation import seasonal_naive_insample_mae

    y = pd.Series(list(range(12)) + [v + 10 for v in range(12)])
    assert seasonal_naive_insample_mae(y, season=12) == pytest.approx(10.0)


def test_mase_uses_train_scale_not_test():
    from forecasting.evaluation import mase

    train = pd.Series(list(range(12)) + [v + 10 for v in range(12)])  # escala = 10
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([105.0, 195.0])  # MAE = 5
    assert mase(y_true, y_pred, train, season=12) == pytest.approx(0.5)


def test_mae_and_mape_basic():
    from forecasting.evaluation import mae, mape

    yt = pd.Series([100.0, 200.0, 400.0])
    yp = pd.Series([110.0, 180.0, 400.0])
    assert mae(yt, yp) == pytest.approx((10 + 20 + 0) / 3)
    assert mape(yt, yp) == pytest.approx((10 + 10 + 0) / 3)


def test_deflate_identity_at_base_month():
    """Real-data smoke test for the IPCA base-month invariant."""
    from forecasting.config import load_config
    from forecasting.eda import deflate_by_ipca
    from forecasting.io import load_monthly_series

    try:
        cfg = load_config()
        df = load_monthly_series(cfg)
        defl = deflate_by_ipca(df, base_month=cfg.ipca_base_month)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"real data unavailable: {exc}")
    base = cfg.ipca_base_month
    mask = pd.to_datetime(df["date"]).dt.strftime("%Y-%m") == base
    nominal = pd.to_numeric(df.loc[mask, "iptu"], errors="coerce").to_numpy()
    real = pd.to_numeric(defl.loc[mask, "iptu"], errors="coerce").to_numpy()
    np.testing.assert_allclose(nominal, real, rtol=1e-9, equal_nan=True)


def test_deflate_identity_synthetic(tmp_path):
    """Synthetic test for the IPCA base-month invariant."""
    from forecasting.eda import deflate_by_ipca

    (tmp_path / "data").mkdir()
    ipca = pd.DataFrame({
        "year": [2025, 2025, 2025], "month": [1, 2, 3],
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "ipca_var_pct": [0.0, 10.0, 10.0],
        "ipca_index": [100.0, 110.0, 121.0],
        "deflator_to_2025_12": [1.0, 1.0, 1.0],
    })
    ipca.to_csv(tmp_path / "data" / "ipca_sgs433.csv", index=False)

    df = pd.DataFrame({
        "cod_ibge": [1, 1, 1],
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "iptu": [1000.0, 1000.0, 1000.0],
    })
    base = "2025-02"  # I_base = 110
    defl = deflate_by_ipca(df, base_month=base, analysis_root=tmp_path)

    assert defl.loc[1, "iptu"] == pytest.approx(1000.0)
    assert defl.loc[0, "iptu"] == pytest.approx(1000.0 * 110.0 / 100.0)
    assert defl.loc[2, "iptu"] == pytest.approx(1000.0 * 110.0 / 121.0)


def test_impute_anomalous_year_uses_adjacent_mean():
    from forecasting.eda import impute_anomalous_year

    idx = pd.date_range("2015-01-01", periods=36, freq="MS")
    vals = np.concatenate([np.full(12, 100.0), np.full(12, 10.0), np.full(12, 200.0)])
    s = pd.Series(vals, index=idx)
    out = impute_anomalous_year(s, 2016)
    got = out[out.index.year == 2016].to_numpy()
    np.testing.assert_allclose(got, np.full(12, 150.0))
    assert out[out.index.year == 2015].eq(100.0).all()
    assert out[out.index.year == 2017].eq(200.0).all()


def test_naive_seasonal_repeats_last_cycle():
    from forecasting import models as M

    idx = pd.date_range("2015-01-01", periods=36, freq="MS")
    s = pd.Series(np.arange(36, dtype=float), index=idx)
    fm = M.fit_naive_seasonal(s)
    fc = M.forecast(fm, 12)
    np.testing.assert_allclose(fc.to_numpy(), np.arange(24, 36, dtype=float))
