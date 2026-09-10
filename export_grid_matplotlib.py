"""Render the paper figures from the synthetic runs.

Loading and labelling belong to ``runs``; this script only draws. What it
selects is stated once, in the constants below, and
``tests/test_renderers.py`` asserts every name in them exists in the data —
the guard that was missing when ``"X"`` and ``"MS"`` quietly filtered away two
base quantifiers the sweep records as ``"TX"`` and ``"TMS"``.
"""

import numpy as np
import matplotlib.pyplot as plt

import runs

#: What the published grid shows. A subset by choice, not by accident.
PUBLISHED_BASE_QUANTIFIERS = ("TAC", "TMAX", "T50", "HDy", "TX", "TMS", "DyS", "SMM")

#: The three method simulators, which is the comparison the study is about.
#: The sweep also runs the ``all`` arm, whose meta-quantifier chooses among
#: every simulator's candidates and the real reference scores besides
#: (ADR-0010). It is left out of these figures by choice and not by accident:
#: it is in the runs, and drawing it is a decision about what the paper
#: compares rather than a filter to repair.
PUBLISHED_METHOD_SIMULATORS = runs.SIMULATORS

REFERENCE_SIMULATOR = runs.UNIFORM
REFERENCE_MERGING_FACTORS = (0.25, 0.5, 0.75)
BAG_SIMULATORS = runs.SIMULATORS

QUANTIFIER_MARKERS = {
    "TAC": "o",
    "TMAX": "s",
    "T50": "^",
    "HDy": "D",
    "TX": "P",
    "TMS": "X",
    "DyS": "v",
    "SMM": "*",
}

METHOD_SIMULATOR_COLOURS = {
    runs.UNIFORM: "#1565C0",
    runs.MVN: "#2E7D32",
    runs.DIRICHLET: "#6A1B9A",
}
METHOD_SIMULATOR_LINESTYLES = {
    runs.UNIFORM: "-",
    runs.MVN: "--",
    runs.DIRICHLET: ":",
}

OUTPUT_PNG = "results/grid_matplotlib_stylized.png"
OUTPUT_PDF = "results/grid_matplotlib_stylized.pdf"
OUTPUT_BOXPLOT_PNG = "results/boxplot_mtr_backline.png"
OUTPUT_BOXPLOT_PDF = "results/boxplot_mtr_backline.pdf"
OUTPUT_BOXPLOT_DIST_PNG = "results/boxplot_distribution_xaxis.png"
OUTPUT_BOXPLOT_DIST_PDF = "results/boxplot_distribution_xaxis.pdf"


def load_published_runs():
    """The synthetic runs the figures draw."""
    published = runs.load_labelled(runs.SYNTHETIC)
    return published[
        published["base_quantifier"].isin(PUBLISHED_BASE_QUANTIFIERS)
        & published["method_simulator"].isin(PUBLISHED_METHOD_SIMULATORS)
    ].copy()


def mean_error_per_cell(published):
    """Mean error per method and cell of the grid."""
    return (
        published.groupby(
            [
                "reference_merging_factor",
                "reference_simulator",
                "bag_simulator",
                "method_simulator",
                "base_quantifier",
                "method",
                "bag_merging_factor",
            ],
            observed=True,
        )["absolute_error"]
        .mean()
        .reset_index()
    )


def plot_grid(agg) -> None:
    eps = 1e-9
    x_axis_label_size = 44
    y_axis_label_size = 30
    tick_label_size = 30
    title_size = 30
    legend_font_size = 34
    legend_columns = 4

    plt.style.use("default")
    fig, axes = plt.subplots(3, 3, figsize=(28, 20), facecolor="white")

    legend_handles = {}

    for row_idx, bag_simulator in enumerate(BAG_SIMULATORS):
        for col_idx, reference_merging_factor in enumerate(REFERENCE_MERGING_FACTORS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            cell = agg[
                (
                    np.abs(agg["reference_merging_factor"] - reference_merging_factor)
                    < eps
                )
                & (agg["reference_simulator"] == REFERENCE_SIMULATOR)
                & (agg["bag_simulator"] == bag_simulator)
            ]

            for method in sorted(cell["method"].unique()):
                sub = cell[cell["method"] == method].sort_values("bag_merging_factor")
                if sub.empty:
                    continue
                base_quantifier = str(sub["base_quantifier"].iloc[0])
                method_simulator = str(sub["method_simulator"].iloc[0])
                line = ax.plot(
                    sub["bag_merging_factor"],
                    sub["absolute_error"],
                    marker=QUANTIFIER_MARKERS.get(base_quantifier, "o"),
                    linewidth=2.4,
                    markersize=5.5,
                    color=METHOD_SIMULATOR_COLOURS[method_simulator],
                    linestyle=METHOD_SIMULATOR_LINESTYLES[method_simulator],
                    label=method,
                    alpha=0.95,
                )[0]
                if method not in legend_handles:
                    legend_handles[method] = line

            ax.set_ylim(0.0, 0.45)
            ax.axvline(
                x=reference_merging_factor,
                color="#111827",
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
            )
            ax.grid(True, linestyle="--", alpha=0.25)

            ax.set_xlabel(
                r"$\mathbf{m}_{\mathbf{ts}}$",
                fontsize=x_axis_label_size,
                fontweight="bold",
                labelpad=10,
            )
            if col_idx == 0:
                ax.set_ylabel("MAE", fontsize=y_axis_label_size, fontweight="bold")

            ax.tick_params(axis="both", labelsize=tick_label_size, width=1.8, length=7)
            ax.set_title(
                f"{runs.SIMULATOR_LABELS[bag_simulator]} | "
                + r"$m_{tr}$"
                + f"={reference_merging_factor}",
                fontsize=title_size,
                fontweight="bold",
                pad=10,
            )

    ordered_labels = sorted(legend_handles.keys())
    fig.legend(
        [legend_handles[k] for k in ordered_labels],
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995, 0.96, 0.0),
        mode="expand",
        ncol=legend_columns,
        fontsize=legend_font_size,
        markerscale=2.0,
        handlelength=2.8,
        handletextpad=0.7,
        columnspacing=0.9,
        labelspacing=0.8,
        borderaxespad=0.0,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.82])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _boxplot_runs(published):
    """Only bags no harder than the reference they are matched against."""
    eps = 1e-9
    return published[
        (published["bag_merging_factor"] - published["reference_merging_factor"]) <= eps
    ]


def _draw_grouped_boxplot(ax, box_runs, group_column, groups):
    """One box per (group, method simulator), grouped along the x axis."""
    base_positions = np.arange(len(groups), dtype=float)
    span = 0.66
    width = span / len(PUBLISHED_METHOD_SIMULATORS)
    offsets = np.linspace(
        -span / 2 + width / 2, span / 2 - width / 2, len(PUBLISHED_METHOD_SIMULATORS)
    )
    legend_handles = {}

    for method_simulator, offset in zip(PUBLISHED_METHOD_SIMULATORS, offsets):
        data = []
        positions = []
        for i, group in enumerate(groups):
            values = box_runs[
                (box_runs[group_column] == group)
                & (box_runs["method_simulator"] == method_simulator)
            ]["absolute_error"].dropna().values
            if values.size == 0:
                continue
            data.append(values)
            positions.append(base_positions[i] + offset)

        if not data:
            continue

        colour = METHOD_SIMULATOR_COLOURS[method_simulator]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.7),
            whiskerprops=dict(color=colour, linewidth=1.4),
            capprops=dict(color=colour, linewidth=1.4),
            boxprops=dict(color=colour, linewidth=1.6),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.25)

        legend_handles[runs.SIMULATOR_LABELS[method_simulator]] = bp["boxes"][0]

    ax.set_xticks(base_positions)
    ax.set_ylabel("MAE", fontsize=22, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(-0.01, 0.27)
    return legend_handles


def plot_boxplot_by_reference_merging_factor(published) -> None:
    box_runs = _boxplot_runs(published)
    factors = sorted(box_runs["reference_merging_factor"].unique().tolist())

    fig, ax = plt.subplots(figsize=(24, 10), facecolor="white")
    ax.set_facecolor("white")

    legend_handles = _draw_grouped_boxplot(
        ax, box_runs, "reference_merging_factor", factors
    )
    ax.set_xticklabels([f"{factor:.2f}" for factor in factors], fontsize=15)
    ax.set_xlabel(r"$m_{tr}$", fontsize=22, fontweight="bold")

    ax.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(legend_handles),
        fontsize=19,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(OUTPUT_BOXPLOT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_BOXPLOT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_boxplot_by_bag_simulator(published) -> None:
    box_runs = _boxplot_runs(published)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_facecolor("white")

    legend_handles = _draw_grouped_boxplot(
        ax, box_runs, "bag_simulator", BAG_SIMULATORS
    )
    ax.set_xticklabels(
        [runs.SIMULATOR_LABELS[simulator] for simulator in BAG_SIMULATORS], fontsize=16
    )
    ax.set_xlabel("Bag simulator", fontsize=22, fontweight="bold")

    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(legend_handles),
        fontsize=19,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
    fig.savefig(OUTPUT_BOXPLOT_DIST_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_BOXPLOT_DIST_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    published_runs = load_published_runs()
    plot_grid(mean_error_per_cell(published_runs))
    plot_boxplot_by_reference_merging_factor(published_runs)
    plot_boxplot_by_bag_simulator(published_runs)
    print(
        "Arquivos gerados: "
        f"{OUTPUT_PNG}, {OUTPUT_PDF}, {OUTPUT_BOXPLOT_PNG}, {OUTPUT_BOXPLOT_PDF}, "
        f"{OUTPUT_BOXPLOT_DIST_PNG} e {OUTPUT_BOXPLOT_DIST_PDF}"
    )
