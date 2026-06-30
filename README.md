# revenue-forecasting

Reproducible forecasting experiments for Brazilian municipal tax revenue.

This repository contains the analysis package used in Daniel Jackson Cavalcante
Costa's undergraduate thesis at UESC. It evaluates monthly and annual forecasts
for IPTU and ISSQN revenue in Bahia municipalities, generates tables and figures
for the LaTeX manuscript, and keeps the canonical forecast caches used to audit
the reported results.

## Related repositories

- [`danieljcksn/siconfi-collector`](https://github.com/danieljcksn/siconfi-collector):
  collects and normalizes SICONFI/RREO data, including monthly revenue series
  and the official municipal forecast benchmark.
- [`danieljcksn/revenue-forecasting`](https://github.com/danieljcksn/revenue-forecasting):
  this repository, responsible for modeling, evaluation, robustness checks, and
  manuscript artifacts.

The separation is intentional: `siconfi-collector` is the data engineering
artifact; `revenue-forecasting` consumes its outputs and implements the
forecasting benchmark.

## Scope

The detailed study covers:

- 3 municipalities: Salvador, Camacari, and Ilheus.
- 2 taxes: IPTU and ISSQN.
- 2 horizons: one month ahead and one fiscal year ahead.
- Metrics: MAE, MAPE, and MASE.
- Validation: rolling-origin evaluation with an expanding training window.
- External comparisons: the official municipal forecast and Oliveira (2024).

The manuscript reports a six-method portfolio cached in `data/forecasts/`:
Naive seasonal, ETS, SARIMA, Prophet, Theta, and a simple Ensemble. The current
Python package also keeps the original four-model training driver
(`scripts/run_pipeline.py`) for reproducibility of the core workflow; tables and
figures are generated from the canonical cache.

## Layout

```text
revenue-forecasting/
├── data/
│   ├── ipca_sgs433.csv
│   └── forecasts/              # canonical forecast caches used by the thesis
├── notebooks/                  # thin executable notebooks
├── scripts/
│   ├── run_pipeline.py          # four-model core training driver
│   └── build_tex_artifacts.py   # regenerates manuscript tables and figures
├── src/forecasting/             # reusable analysis package
└── tests/                       # methodological and smoke tests
```

## Setup

Install the collector first, then this package:

```bash
pip install -e ../siconfi-collector
pip install -e ".[dev]"
```

The optional `precisao` extra records the dependency used by the canonical
six-method run:

```bash
pip install -e ".[precisao]"
```

For fully pinned environments, use `requirements-lock.txt` or
`requirements-sf-lock.txt`.

## Usage

Generate every table and figure from the cached forecasts:

```bash
python scripts/build_tex_artifacts.py --all
```

Run the original four-model training driver:

```bash
python scripts/run_pipeline.py
```

The scripts expect the thesis wrapper configuration file `.tcc-pipeline.json`,
which defines the manuscript root, SICONFI data location, forecast cache
directory, sample window, municipalities, taxes, and output paths.

## Validation

```bash
python -m ruff check src tests scripts
python -m pytest
```

The tests cover the main methodological risks: MASE denominator, IPCA deflation,
known anomaly imputation, rolling-origin leakage, annual aggregation, model
factory structure, and smoke tests for installed model backends.
