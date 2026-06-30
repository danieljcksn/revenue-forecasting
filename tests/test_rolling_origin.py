"""Synthetic tests for rolling-origin validation.

They cover leakage prevention, monthly-to-annual aggregation, expected fold
counts, and NumPy seed reproducibility. The tests use only NumPy, pandas, and
the deterministic seasonal Naive model so that validation logic is isolated
from heavier model backends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import models as M
from forecasting.benchmarks import aggregate_monthly_to_annual

# Same hyperparameters used by the rolling-origin pipeline.
INITIAL_WINDOW = 72
MAX_HORIZON = 12
STEP = 1
SEASON = 12


def _deterministic_series(n: int = 96, start: str = "2015-01-01") -> pd.Series:
    """Serie mensal determinÃ­stica, estritamente positiva, com sazonalidade.

    Sem componente aleatorio: nivel + tendencia linear + sazonalidade anual.
    O determinismo e essencial para as comparacoes bit a bit deste modulo.
    """
    idx = pd.date_range(start, periods=n, freq="MS")
    t = np.arange(n, dtype=float)
    seasonal = 10.0 * np.sin(2.0 * np.pi * (t % SEASON) / SEASON)
    y = 1000.0 + 5.0 * t + seasonal  # > 0 em todo o dominio
    return pd.Series(y, index=idx, name="y")


class _SpyFitter:
    """Embrulha um ``fit_fn`` e registra a serie de treino recebida por dobra.

    Permite inspecionar EXATAMENTE quais observacoes ``rolling_origin_cv``
    entrega ao modelo em cada origem, sem alterar o comportamento (delega para
    o fitter interno).
    """

    def __init__(self, inner):
        self._inner = inner
        self.seen_trains: list[pd.Series] = []

    def __call__(self, train: pd.Series):
        self.seen_trains.append(train.copy())  # copia defensiva
        return self._inner(train)


# ---------- 1a. Vazamento: prova estrutural -------------------------------


def test_each_fold_train_is_strictly_in_the_past():
    """Cada dobra treina somente no prefixo anterior a origem (sem vazamento)."""
    y = _deterministic_series(n=96)
    spy = _SpyFitter(M.fit_naive_seasonal)
    cv = M.rolling_origin_cv(
        y, spy, initial_window=INITIAL_WINDOW, max_horizon=MAX_HORIZON, step=STEP,
    )

    # Origens esperadas: 72..95 (janela expansiva, passo 1), todas com >=1 passo.
    expected_origins = list(range(INITIAL_WINDOW, len(y)))
    assert len(spy.seen_trains) == len(expected_origins)

    for o, train in zip(expected_origins, spy.seen_trains):
        origin_date = y.index[o]  # primeira data-alvo (ja no futuro do treino)
        # (a) janela expansiva: exatamente 'o' observacoes no treino
        assert len(train) == o
        # (b) TODA observacao de treino e anterior a origem
        assert train.index.max() < origin_date
        assert bool((train.index < origin_date).all())
        # (c) o treino e o prefixo exato da serie (mesmos valores e datas)
        assert np.array_equal(train.to_numpy(), y.to_numpy()[:o])
        assert train.index.equals(y.index[:o])

    # No DataFrame retornado: train_end < target_date em TODA linha.
    assert bool((cv["train_end"] < cv["target_date"]).all())
    # E train_end e sempre a ultima data do treino daquela origem.
    for o in expected_origins:
        sub = cv[cv["train_end"] == y.index[o - 1]]
        assert not sub.empty
        assert bool((sub["target_date"] > y.index[o - 1]).all())


# ---------- 1c. Vazamento: oraculo indice<->alvo + contagem de dobras -------


def test_fold_counts_and_target_index_oracle():
    """Oraculo alvo<->indice e numero de dobras por horizonte.

    Codifica o indice temporal NO VALOR (``y[i] = i``): assim cada ``y_true``
    tem de ser exatamente o indice posicional da sua data-alvo -- um
    desalinhamento ou vazamento na montagem das dobras quebraria o oraculo.
    Fixa tambem o numero de dobras esperado em ``h=1`` e ``h=12`` para um ``n``
    conhecido (janela inicial 72, passo 1, horizonte ate 12).
    """
    n = 96
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    y = pd.Series(np.arange(n, dtype=float), index=idx, name="y")  # valor == indice
    cv = M.rolling_origin_cv(
        y, M.fit_naive_seasonal,
        initial_window=INITIAL_WINDOW, max_horizon=MAX_HORIZON, step=STEP,
    )

    # (a) oraculo: y_true de cada linha == indice posicional da data-alvo.
    pos = {d: i for i, d in enumerate(idx)}
    assert all(int(r.y_true) == pos[r.target_date] for r in cv.itertuples())
    # (b) o alvo e SEMPRE estritamente futuro frente ao fim do treino.
    assert bool((cv["target_date"] > cv["train_end"]).all())
    # (c) numero de dobras por horizonte (n=96, janela 72, passo 1):
    #     h=1 existe em toda origem 72..95            -> 24 dobras;
    #     h=12 so nas origens 72..84 (restam >=12)    -> 13 dobras.
    assert cv["origin"].nunique() == n - INITIAL_WINDOW                       # 24 origens
    assert int((cv["step"] == 1).sum()) == n - INITIAL_WINDOW                 # 24
    assert int((cv["step"] == MAX_HORIZON).sum()) == n - INITIAL_WINDOW - MAX_HORIZON + 1  # 13
    # (d) passos por origem e total de linhas (soma de min(12, n-o)).
    first_origin_rows = int((cv["origin"] == idx[INITIAL_WINDOW - 1]).sum())
    penultimate_origin_rows = int((cv["origin"] == idx[n - 2]).sum())
    assert first_origin_rows == MAX_HORIZON  # 1a origem: 12 passos
    assert penultimate_origin_rows == 1  # ultima origem: 1 passo
    assert len(cv) == 13 * MAX_HORIZON + sum(range(1, MAX_HORIZON))            # 156 + 66 = 222


# ---------- 1b. Vazamento: prova comportamental (sentinela) ----------------


def test_future_sentinel_does_not_change_past_origin_forecasts():
    """Sentinela no futuro nao muda as previsoes das dobras de treino intacto.

    Se a previsao do passo h espiasse qualquer valor posterior a origem, um
    sentinela absurdo no futuro mudaria as previsoes -- aqui provamos que nao
    muda, bit a bit, para toda dobra cujo treino esta no passado nao corrompido.
    """
    y = _deterministic_series(n=96)
    clean = M.rolling_origin_cv(
        y, M.fit_naive_seasonal,
        initial_window=INITIAL_WINDOW, max_horizon=MAX_HORIZON, step=STEP,
    )

    # Corta em 84: observacoes 0..83 intactas; 84..95 viram sentinela gigante.
    cut = 84
    cut_date = y.index[cut - 1]
    poisoned = y.copy()
    poisoned.iloc[cut:] = 1e12

    corrupt = M.rolling_origin_cv(
        poisoned, M.fit_naive_seasonal,
        initial_window=INITIAL_WINDOW, max_horizon=MAX_HORIZON, step=STEP,
    )

    # Dobras cujo treino esta inteiramente no passado intacto: train_end <= cut.
    a = clean.loc[clean["train_end"] <= cut_date].reset_index(drop=True)
    b = corrupt.loc[corrupt["train_end"] <= cut_date].reset_index(drop=True)

    assert len(a) > 0 and len(a) == len(b)  # sanidade: tais dobras existem
    # Previsao e escala in-sample IDENTICAS bit a bit -> sem vazamento do futuro.
    np.testing.assert_array_equal(a["y_pred"].to_numpy(), b["y_pred"].to_numpy())
    np.testing.assert_array_equal(
        a["insample_scale"].to_numpy(), b["insample_scale"].to_numpy()
    )


# ---------- 2. Agregacao mensal -> anual (origem-dezembro, n==12) ----------


def test_aggregate_monthly_to_annual_december_origin_full_year():
    """So origens de dezembro com os 12 passos completos viram previsao anual."""
    rows: list[dict] = []
    base = dict(municipio="ilheus", municipio_nome="IlhÃ©us", tributo="IPTU",
                modelo="SARIMA", insample_scale=1.0)

    # (i) Dezembro/2020 COMPLETO (12 passos) -> ano-alvo 2021. Deve sobreviver.
    dec_origin = pd.Timestamp("2020-12-01")
    preds = [100.0 + i for i in range(12)]  # soma = 1266
    trues = [110.0 + i for i in range(12)]  # soma = 1386
    for k in range(12):
        rows.append({**base, "origin": dec_origin, "step": k + 1,
                     "target_date": dec_origin + pd.offsets.MonthBegin(k + 1),
                     "y_true": trues[k], "y_pred": preds[k]})

    # (ii) Junho/2021 (nao-dezembro): deve ser IGNORADO pelo filtro de mes.
    jun_origin = pd.Timestamp("2021-06-01")
    for k in range(12):
        rows.append({**base, "origin": jun_origin, "step": k + 1,
                     "target_date": jun_origin + pd.offsets.MonthBegin(k + 1),
                     "y_true": 1.0, "y_pred": 999.0})

    # (iii) Dezembro/2022 INCOMPLETO (6 passos): descartado por n_steps != 12.
    dec_incomplete = pd.Timestamp("2022-12-01")
    for k in range(6):
        rows.append({**base, "origin": dec_incomplete, "step": k + 1,
                     "target_date": dec_incomplete + pd.offsets.MonthBegin(k + 1),
                     "y_true": 1.0, "y_pred": 1.0})

    cv = pd.DataFrame(rows)
    assert cv["origin"].dtype.kind == "M"  # garante datetime para .dt.month

    annual = aggregate_monthly_to_annual(cv)

    # Apenas o bloco de dezembro/2020 completo sobrevive.
    assert len(annual) == 1
    row = annual.iloc[0]
    assert int(row["target_year"]) == 2021
    assert int(row["n_steps"]) == 12
    assert row["pred_annual"] == pytest.approx(sum(preds))
    assert row["real_annual"] == pytest.approx(sum(trues))
    expected_err = 100.0 * abs(sum(preds) - sum(trues)) / abs(sum(trues))
    assert row["err_pct_model"] == pytest.approx(expected_err)


# ---------- 3. Reprodutibilidade: set_global_seeds fixa o RNG do numpy ------


def test_set_global_seeds_makes_numpy_reproducible():
    """``set_global_seeds`` torna o RNG do numpy determinÃ­stico (mesma seed)."""
    M.set_global_seeds(42)
    a = np.random.rand(8)
    M.set_global_seeds(42)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)

    # Seeds diferentes -> sequencias diferentes (sanidade).
    M.set_global_seeds(123)
    c = np.random.rand(8)
    assert not np.array_equal(a, c)

