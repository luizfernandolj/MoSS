"""Report the real-data ranking as a statistical claim, not a table of means (#12).

Loading and labelling belong to ``runs``; the comparison itself to ``stats``.
This script's own job is the one left after those two: name the headline
pair, draw the picture, print the numbers — the same split
``export_grid_matplotlib.py`` draws between ``grid.panels`` and its own
plotting calls.
"""

import matplotlib.pyplot as plt

import runs
import stats

#: The pair CONTEXT.md itself uses to introduce "method": the same base
#: quantifier, with and without an MVN-simulated reference standing in for
#: the real training scores. Naming this pair is a decision about what the
#: report's headline claim is about, not a default inferred from whichever
#: methods happen to rank first today.
HEADLINE_PAIR = ("DyS", "QuaDapt-MVN(DyS)")

OUTPUT_CD_DIAGRAM_PNG = "results/critical_difference_diagram.png"
OUTPUT_CD_DIAGRAM_PDF = "results/critical_difference_diagram.pdf"


def load_real_data_runs():
    """The real-data runs the report compares: every method the table names, not a published subset."""
    return runs.load_labelled(runs.REAL_DATA)


def critical_difference_diagram(nemenyi_result, ax=None):
    """Demsar's (2006) diagram: methods placed by average rank, a bar under every clique.

    Draws exactly what ``nemenyi_result`` already decided
    (:func:`stats.nemenyi`) — grouping methods into cliques is that
    function's job, not this one's.
    """
    ranks = nemenyi_result.average_ranks
    methods = list(ranks.index)
    values = ranks.to_numpy()
    n = len(methods)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(9, 0.5 * n), 0.35 * n + 1.5))

    ax.set_xlim(1 - 0.5, n + 0.5)
    ax.set_ylim(-0.5 - 0.25 * len(nemenyi_result.cliques), n + 1)
    ax.set_xticks(range(1, n + 1))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Average rank (lower is better)")
    for spine in ("left", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])

    for position, (method, rank) in enumerate(zip(methods, values)):
        y = n - position
        ax.plot([rank, rank], [0, y], color="#9CA3AF", linewidth=1, zorder=1)
        ax.plot(rank, 0, marker="o", color="#374151", zorder=2)
        ax.text(
            rank,
            y + 0.15,
            f"{method} ({rank:.2f})",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=60,
        )

    bar_y = -0.3
    for clique in nemenyi_result.cliques:
        clique_ranks = ranks[list(clique)]
        ax.plot(
            [clique_ranks.min(), clique_ranks.max()],
            [bar_y, bar_y],
            color="#111827",
            linewidth=3,
            solid_capstyle="butt",
        )
        bar_y -= 0.25

    ax.set_title(
        f"Critical difference = {nemenyi_result.critical_difference:.3f} "
        f"(alpha={nemenyi_result.alpha}, n={nemenyi_result.n_datasets} datasets)"
    )
    return ax


def report(labelled_runs):
    """Print and draw the three things #12 asks for, from one already-loaded table."""
    friedman_result = stats.friedman(labelled_runs)
    nemenyi_result = stats.nemenyi(labelled_runs)
    headline = stats.pairwise_signed_rank(labelled_runs, *HEADLINE_PAIR)

    print(
        f"Friedman: chi2={friedman_result.statistic:.3f} "
        f"p={friedman_result.p_value:.3e} "
        f"(n_datasets={friedman_result.n_datasets}, n_methods={friedman_result.n_methods})"
    )
    print(
        f"Nemenyi critical difference (alpha={nemenyi_result.alpha}): "
        f"{nemenyi_result.critical_difference:.3f}"
    )
    print(
        f"{headline.method_a} vs {headline.method_b} (signed-rank): "
        f"W={headline.statistic:.3f} p={headline.p_value:.3e} "
        f"(n_datasets={headline.n_datasets})"
    )

    ax = critical_difference_diagram(nemenyi_result)
    fig = ax.figure
    fig.savefig(OUTPUT_CD_DIAGRAM_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_CD_DIAGRAM_PDF, bbox_inches="tight")
    plt.close(fig)

    return friedman_result, nemenyi_result, headline


if __name__ == "__main__":
    report(load_real_data_runs())
    print(f"Wrote {OUTPUT_CD_DIAGRAM_PNG} and {OUTPUT_CD_DIAGRAM_PDF}")
