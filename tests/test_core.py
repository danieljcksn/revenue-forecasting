"""Core tests for metrics, deflation, anomaly treatment, and aggregation.

These tests protect invariants that would silently alter generated tables and
figures if broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------- MASE: denominador in-sample do Naive sazonal -----------------


def test_seasonal_naive_insample_mae_known_value():
    from forecasting.evaluation import seasonal_naive_insample_mae

    # serie de 24 meses; diferenca sazonal (lag 12) constante = 10 -> MAE = 10
    y = pd.Series(list(range(12)) + [v + 10 for v in range(12)])
    assert seasonal_naive_insample_mae(y, season=12) == pytest.approx(10.0)


def test_mase_uses_train_scale_not_test():
    from forecasting.evaluation import mase

    train = pd.Series(list(range(12)) + [v + 10 for v in range(12)])  # escala = 10
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([105.0, 195.0])  # MAE = 5
    # MASE = 5 / 10 = 0,5
    assert mase(y_true, y_pred, train, season=12) == pytest.approx(0.5)


def test_mae_and_mape_basic():
    from forecasting.evaluation import mae, mape

    yt = pd.Series([100.0, 200.0, 400.0])
    yp = pd.Series([110.0, 180.0, 400.0])
    assert mae(yt, yp) == pytest.approx((10 + 20 + 0) / 3)
    # APE: 10%, 10%, 0% -> 6,667%
    assert mape(yt, yp) == pytest.approx((10 + 10 + 0) / 3)


# ---------- Deflacao: invariancia no mes-base ----------------------------


def test_deflate_identity_at_base_month():
    """Integracao (OPCIONAL): deflaciona os dados REAIS e confere a invariancia
    no mes-base. E pulado quando os CSVs reais nao estao acessiveis (ex.: maquina
    de desenvolvimento sem o siconfi-collector, ou ``.tcc-pipeline.json`` com
    paths de outra maquina). A invariancia em si e coberta, em qualquer maquina,
    por ``test_deflate_identity_synthetic``.
    """
    from forecasting.config import load_config
    from forecasting.eda import deflate_by_ipca
    from forecasting.io import load_monthly_series

    try:
        cfg = load_config()
        df = load_monthly_series(cfg)
        defl = deflate_by_ipca(df, base_month=cfg.ipca_base_month)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"dados reais indisponiveis nesta maquina: {exc}")
    base = cfg.ipca_base_month
    # No mes-base, valores reais == nominais (fator = 1)
    mask = pd.to_datetime(df["date"]).dt.strftime("%Y-%m") == base
    nominal = pd.to_numeric(df.loc[mask, "iptu"], errors="coerce").to_numpy()
    real = pd.to_numeric(defl.loc[mask, "iptu"], errors="coerce").to_numpy()
    np.testing.assert_allclose(nominal, real, rtol=1e-9, equal_nan=True)


def test_deflate_identity_synthetic(tmp_path):
    """Mesma propriedade da deflacao, mas com IPCA e serie SINTETICOS (roda em
    qualquer maquina): no mes-base o fator e 1 (real == nominal) e, fora dele,
    ``real = nominal * I_base / I_t``.
    """
    from forecasting.eda import deflate_by_ipca

    # IPCA acumulado sintetico: 10% ao mes (2025-01=100, -02=110, -03=121).
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

    # mes-base: fator = 110/110 = 1 -> real == nominal
    assert defl.loc[1, "iptu"] == pytest.approx(1000.0)
    # 2025-01: 1000 * 110/100 = 1100 ; 2025-03: 1000 * 110/121
    assert defl.loc[0, "iptu"] == pytest.approx(1000.0 * 110.0 / 100.0)
    assert defl.loc[2, "iptu"] == pytest.approx(1000.0 * 110.0 / 121.0)


# ---------- Imputacao da anomalia anual ----------------------------------


def test_impute_anomalous_year_uses_adjacent_mean():
    from forecasting.eda import impute_anomalous_year

    idx = pd.date_range("2015-01-01", periods=36, freq="MS")
    vals = np.concatenate([np.full(12, 100.0), np.full(12, 10.0), np.full(12, 200.0)])
    s = pd.Series(vals, index=idx)
    out = impute_anomalous_year(s, 2016)
    # cada mes de 2016 -> media(2015, 2017) = (100 + 200)/2 = 150
    got = out[out.index.year == 2016].to_numpy()
    np.testing.assert_allclose(got, np.full(12, 150.0))
    # anos vizinhos intactos
    assert out[out.index.year == 2015].eq(100.0).all()
    assert out[out.index.year == 2017].eq(200.0).all()


# ---------- Naive sazonal: repete o ultimo ciclo -------------------------


def test_naive_seasonal_repeats_last_cycle():
    from forecasting import models as M

    idx = pd.date_range("2015-01-01", periods=36, freq="MS")
    s = pd.Series(np.arange(36, dtype=float), index=idx)
    fm = M.fit_naive_seasonal(s)
    fc = M.forecast(fm, 12)
    # proximos 12 meses = ultimos 12 observados (24..35)
    np.testing.assert_allclose(fc.to_numpy(), np.arange(24, 36, dtype=float))
