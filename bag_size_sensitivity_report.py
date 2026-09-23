"""Draw the bag-size sensitivity figures from real-data runs (#19, #20).

Loading and labelling belong to ``runs``; shaping the MAE trend and the
per-bag-size rankings belongs to ``bag_size_report``; this script's own job is
the one left after those two — name the figures and save them — the same
split ``statistical_report.py`` draws between ``stats`` and its own plotting
calls, and reuses that module's own critical-difference drawing rather than a
second copy of it.
"""

import matplotlib.pyplot as plt

import bag_size_report
import runs
import statistical_report

OUTPUT_MAE_PNG = "results/mae_by_bag_size.png"
OUTPUT_MAE_PDF = "results/mae_by_bag_size.pdf"

#: One critical-difference diagram per bag size, named so a reader can tell
#: which bag size each belongs to without opening it.
OUTPUT_CD_DIAGRAM_PNG_TEMPLATE = "results/critical_difference_diagram_bag_size_{bag_size}.png"
OUTPUT_CD_DIAGRAM_PDF_TEMPLATE = "results/critical_difference_diagram_bag_size_{bag_size}.pdf"


def load_real_data_runs():
    """Every real-data run, every bag size the sweep produced (#16).

    No filter on replication: a bag a small pool could only fill by
    bootstrap-replicating a short class (#18) is included by default, the
    same reasoning ``bag_size_report.mae_by_bag_size`` and
    ``bag_size_report.rankings_by_bag_size`` follow.
    """
    return runs.load_labelled(runs.REAL_DATA)


def cd_diagram_paths(bag_size):
    """Where one bag size's critical-difference diagram is written."""
    return (
        OUTPUT_CD_DIAGRAM_PNG_TEMPLATE.format(bag_size=bag_size),
        OUTPUT_CD_DIAGRAM_PDF_TEMPLATE.format(bag_size=bag_size),
    )


def plot_mae_by_bag_size(mae_table, datasets=None):
    """One subplot per dataset: mean absolute error against bag size.

    ``mae_table`` is already shaped and averaged
    (:func:`bag_size_report.mae_by_bag_size`) — this function only draws it,
    the same split ``export_grid_matplotlib.plot_grid`` draws against
    ``grid.panels``. ``datasets`` defaults to every dataset the table names,
    sorted, so the figure needs no second list of datasets kept in step with
    the sweep.
    """
    if datasets is None:
        datasets = sorted(mae_table["dataset"].unique())

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        sub = mae_table[mae_table["dataset"] == dataset].sort_values("bag_size")
        ax.plot(sub["bag_size"], sub["absolute_error"], marker="o", color="#1565C0")
        ax.set_xscale("log")
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("Bag size")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("MAE")

    fig.tight_layout()
    fig.savefig(OUTPUT_MAE_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_MAE_PDF, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_rankings_by_bag_size(rankings):
    """One critical-difference diagram per bag size, saved as PNG and PDF each.

    Draws exactly what each ranking's own ``nemenyi`` result decided
    (:func:`statistical_report.critical_difference_diagram`) — grouping
    methods into cliques is that function's job, not this one's, the same
    reasoning it already states for the single-ranking report.
    """
    written = []
    for ranking in rankings:
        ax = statistical_report.critical_difference_diagram(ranking.nemenyi)
        fig = ax.figure
        png_path, pdf_path = cd_diagram_paths(ranking.bag_size)
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        written.append((png_path, pdf_path))
    return written


def report(labelled_runs):
    """Shape and draw both bag-size sensitivity figures from one already-loaded table."""
    mae_table = bag_size_report.mae_by_bag_size(labelled_runs)
    plot_mae_by_bag_size(mae_table)

    rankings = bag_size_report.rankings_by_bag_size(labelled_runs)
    plot_rankings_by_bag_size(rankings)

    return mae_table, rankings


if __name__ == "__main__":
    mae_table, rankings = report(load_real_data_runs())
    print(f"Wrote {OUTPUT_MAE_PNG} and {OUTPUT_MAE_PDF}")
    for ranking in rankings:
        png_path, pdf_path = cd_diagram_paths(ranking.bag_size)
        print(f"Wrote {png_path} and {pdf_path}")
