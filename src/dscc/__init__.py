"""dscc -- Dynamic Spectral Conditional Correlations in Python.

Implements Abadir and Rockinger (2026, Econometrics Journal,
doi:10.1093/ectj/utag017) and the explicit minimal representation of variance
matrices of Abadir (2023, Econometrics Journal 26, 88-104, doi:10.1093/ectj/utac023).

Developed and maintained by Dr Merwan Roudane <merwanroudane920@gmail.com>,
https://github.com/merwanroudane/dscc
"""
from ._filters import dsh_filter, g_abs
from ._model import DSCC, DSCCForecast, DSCCResults, default_start, param_names
from ._representation import (
    compose_sigma,
    correlation_from_cov,
    decompose_sigma,
    orthogonal_from_data,
    permutation_matrix,
    skew_from_vec,
    vec_from_skew,
    xi_from_data,
)

from ._colors import (bluered_colors, get_cmap, matlab_jet_colors, parula_colors, resolve_colorscale,
                      sinha_colors, turbo_colors)
from . import backtest, datasets, dcc, elliptope, portfolio
from ._plots import (plot_correlation_heatmap, plot_coverage_grid, plot_elliptope, plot_slice_coverage,
                     plot_wealth, save_figure)
from .backtest import DCCForecaster, DSCCForecaster, EqualWeight, SampleCovForecaster, rolling_backtest
from .dcc import DCC
from .portfolio import gmv_weights

__version__ = "0.1.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

__all__ = [
    "DSCC", "DSCCResults", "DSCCForecast", "default_start", "param_names",
    "dsh_filter", "g_abs",
    "compose_sigma", "decompose_sigma", "correlation_from_cov", "orthogonal_from_data",
    "permutation_matrix", "skew_from_vec", "vec_from_skew", "xi_from_data",
    "parula_colors", "matlab_jet_colors", "turbo_colors", "bluered_colors", "sinha_colors",
    "resolve_colorscale", "get_cmap",
    "plot_elliptope", "plot_slice_coverage", "plot_coverage_grid", "plot_wealth",
    "plot_correlation_heatmap", "save_figure",
    "DCC", "gmv_weights", "rolling_backtest", "DSCCForecaster", "DCCForecaster",
    "SampleCovForecaster", "EqualWeight",
    "backtest", "datasets", "dcc", "elliptope", "portfolio",
]
