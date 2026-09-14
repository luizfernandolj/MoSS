"""Which runs appear in which panel of the grid. The renderers only draw.

The published figure and the dashboard both show the same 3x3 arrangement —
rows of bag simulator, columns of reference merging factor, one line per
method inside each — and each used to decide that arrangement for itself:
the same eps-tolerant masking, the same splitting-out of the baseline
quantifier into its own line, the same downsampling, written twice and free
to drift. This module writes it once. A renderer's whole job is to call
:func:`panels` and draw what comes back.

Called *panel*, not *cell*: :class:`sweep.Cell` and :class:`real_data.Cell`
already name a point of the *experiment* grid — one way of making the
reference score set and the bags. A panel is a point of the *figure* grid, a
crossing of bag simulator and reference merging factor that a subplot draws.
Reusing "cell" for both would be exactly the kind of silent rename the rest
of this project's vocabulary discipline exists to prevent.

Averaging over repetitions is this module's job too, not the renderers':
each panel's ``methods`` and ``baseline`` frames carry the mean
``absolute_error`` per (method, bag merging factor), which is what a line
plot draws one point from — a renderer that averaged for itself would be the
same duplication one level down.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import runs

_EPS = 1e-9

#: The x axis every panel plots against.
X_COLUMN = "bag_merging_factor"

#: How a run is attributed to a panel, and to a line within it. Averaging
#: collapses everything *not* named here — which is exactly ``repetition``.
_GROUP_COLUMNS = (
    "reference_merging_factor",
    "reference_simulator",
    "bag_simulator",
    "method_simulator",
    "base_quantifier",
    "method",
    X_COLUMN,
)

#: Points a single line carries before a panel downsamples it. Generous
#: relative to the published grid's own density on purpose: downsampling
#: exists to keep an unusually dense sweep readable, not to thin the
#: published one.
MAX_POINTS_PER_METHOD = 1000

#: How the baseline quantifier's line is named, once split out of the
#: methods it would otherwise be indistinguishable from.
BASELINE_LABEL = f"{runs.BASELINE_QUANTIFIER} (baseline)"

#: The grid's shape: which bag simulator is drawn in which row, which
#: reference merging factor in which column. The published figure and the
#: dashboard draw the same shape — the figure fixes which reference
#: simulator and which methods appear, the dashboard lets a reader choose
#: those, but the rows and columns underneath are the one grid, not two that
#: happen to agree. Named here so both import the shape rather than each
#: spelling it out.
BAG_SIMULATORS = runs.SIMULATORS
REFERENCE_MERGING_FACTORS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class Panel:
    """One subplot: a bag simulator crossed with a reference merging factor.

    ``methods`` and ``baseline`` are both ready to draw as-is: aggregated,
    labelled, ordered by method then by :data:`X_COLUMN`, and downsampled.
    Split apart because :data:`runs.BASELINE_QUANTIFIER` reads no
    reference score set and is drawn as a single distinguished line rather
    than as one method among the rest — a renderer that skips the baseline
    (as the published figure does, by excluding it upstream) simply finds
    ``baseline`` empty.
    """

    bag_simulator: str
    reference_merging_factor: float
    methods: pd.DataFrame
    baseline: pd.DataFrame


def panels(
    labelled_runs,
    *,
    reference_simulator,
    bag_simulators,
    reference_merging_factors,
    max_points_per_method=MAX_POINTS_PER_METHOD,
):
    """The grid's panels, each already reduced to what a renderer draws.

    ``labelled_runs`` carries ``method`` and ``absolute_error``
    (:func:`runs.load_labelled`). Which base quantifiers and method
    simulators appear in the grid at all is the caller's choice, made by
    filtering ``labelled_runs`` before calling this function — this function
    only ever places the runs it is given, in the order ``bag_simulators``
    and ``reference_merging_factors`` name.
    """
    aggregated = _aggregate(labelled_runs)
    return tuple(
        _panel(
            aggregated,
            bag_simulator,
            reference_merging_factor,
            reference_simulator,
            max_points_per_method,
        )
        for bag_simulator in bag_simulators
        for reference_merging_factor in reference_merging_factors
    )


def _aggregate(labelled_runs):
    return (
        labelled_runs.groupby(list(_GROUP_COLUMNS), observed=True)["absolute_error"]
        .mean()
        .reset_index()
    )


def _panel(
    aggregated, bag_simulator, reference_merging_factor, reference_simulator, max_points_per_method
):
    mask = (
        (np.abs(aggregated["reference_merging_factor"] - reference_merging_factor) < _EPS)
        & (aggregated["reference_simulator"] == reference_simulator)
        & (aggregated["bag_simulator"] == bag_simulator)
    )
    methods, baseline = _split_out_baseline(aggregated[mask])
    return Panel(
        bag_simulator=bag_simulator,
        reference_merging_factor=reference_merging_factor,
        methods=_ordered(_downsample(methods, max_points_per_method)),
        baseline=_ordered(_downsample(baseline, max_points_per_method)),
    )


def _split_out_baseline(panel_runs):
    """Return (``panel_runs`` without the baseline, the baseline's own rows).

    The baseline is the one base quantifier with no method simulator
    (:data:`runs.BASELINE_QUANTIFIER`, :data:`runs.NO_METHOD_SIMULATOR`): it
    reads no reference score set, so it is not a line among the others but
    the one every other line is compared against.
    """
    is_baseline = (panel_runs["base_quantifier"] == runs.BASELINE_QUANTIFIER) & (
        panel_runs["method_simulator"] == runs.NO_METHOD_SIMULATOR
    )
    baseline = panel_runs[is_baseline].copy()
    if not baseline.empty:
        baseline["method"] = BASELINE_LABEL
    return panel_runs[~is_baseline].copy(), baseline


def _downsample(frame, max_points_per_method):
    """At most ``max_points_per_method`` points per line, spread across it.

    A no-op on the published grid, whose panels hold a handful of merging
    factors per line — the guard exists for a sweep dense enough that a line
    would otherwise carry thousands of points.
    """
    if frame.empty:
        return frame
    kept = []
    for method in frame["method"].unique():
        sub = frame[frame["method"] == method]
        if len(sub) > max_points_per_method:
            sub = sub.sort_values(X_COLUMN).iloc[
                np.linspace(0, len(sub) - 1, max_points_per_method).astype(int)
            ]
        kept.append(sub)
    return pd.concat(kept, ignore_index=True)


def _ordered(frame):
    """Rows sorted by method then by x, so every renderer draws the same line order."""
    if frame.empty:
        return frame
    return frame.sort_values(["method", X_COLUMN]).reset_index(drop=True)
