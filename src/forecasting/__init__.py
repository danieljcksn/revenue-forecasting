"""Reusable forecasting package for municipal tax-revenue experiments.

The package consumes normalized SICONFI/RREO outputs produced by
``siconfi-collector`` and provides the analysis layer: series preparation,
model wrappers, rolling-origin evaluation, external benchmarks, tables, and
figures. The reported portfolio contains Naive seasonal, ETS, SARIMA, Prophet,
Theta, and a simple Ensemble over monthly and annual horizons.
"""

from forecasting.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
