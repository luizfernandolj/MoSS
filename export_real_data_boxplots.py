"""Render the real-data ranking boxplot per bag size, meta-quantifiers only.

``dashboard.py`` draws this interactively with Plotly; this draws the same
panel — ``stats.dataset_method_ranks``, one box per method over the datasets —
with matplotlib, at a size and weight meant for a paper figure or a slide.
Matplotlib rather than Plotly because Plotly needs ``kaleido`` to write an
image and that is not installed, and because the other published figures
already use it (``export_grid_matplotlib.py``).

What it selects is the meta-quantifier arms alone: every method whose
``method_simulator`` is not :data:`runs.NO_METHOD_SIMULATOR`. That drops the
eleven plain quantifiers — ``DyS``, ``HDy``, ``SORD``, ``SMM``, the six
threshold-policy ones and the ``CC`` baseline — which read no reference
substitute at all and so answer a different question than the one a figure
comparing simulators asks.

Boxes run horizontally because the labels are long (``QuaDapt-Dirichlet(TMS2)``)
and there are forty of them: rotated vertical ticks at a readable point size
would take more height than the plot. Best rank sits at the top.

One figure per bag size rather than one pooled figure, the same split
``bag_size_report.rankings_by_bag_size`` makes (#20): one method at two bag
sizes is two measurements of different things, and pooling them would rank it
against itself.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bag_size_report
import runs
import stats

#: One colour per method simulator, the first shade of ``dashboard.py``'s own
#: palette for it — so a method is the same colour here as in the dashboard
#: rather than in a second scheme that drifts from it. MVN is darkened from
#: the dashboard's ``#00FF00``, which is legible on a screen and vanishes on
#: a projector and in print.
METHOD_SIMULATOR_COLOURS = {
    runs.UNIFORM: "#0047AB",
    runs.MVN: "#00A000",
    runs.DIRICHLET: "#8B00FF",
    runs.ALL_SIMULATORS: "#EA580C",
    #: The plain quantifiers and the CC baseline — no meta-quantifier, no
    #: simulated reference. Grey, as the dashboard has them: they are the
    #: thing the four coloured arms are measured against rather than a fifth
    #: arm beside them.
    runs.NO_METHOD_SIMULATOR: "#6B7280",
}

#: Spelled for a reader rather than reusing the registry key, since these go
#: in a legend someone reads from the back of a room. Portuguese, like
#: ``dashboard.py``'s own labels — these figures are for a paper and slides
#: written in it, while the docs and the code stay in English.
METHOD_SIMULATOR_LABELS = {
    runs.UNIFORM: "Uniforme",
    runs.MVN: "MVN",
    runs.DIRICHLET: "Dirichlet",
    #: "All", not "Todos" and not spelled out as "Todos + referência real":
    #: this names the arm, and the arm is what the tick labels already call
    #: ``QuaDapt-All(...)``. That the real reference scores are among its
    #: candidates (ADR-0010) is a fact about the method, documented where the
    #: method is, rather than something a legend swatch has room to say.
    runs.ALL_SIMULATORS: "All",
    runs.NO_METHOD_SIMULATOR: "Sem meta-quantificador",
}

#: Every string the figure shows, in one place rather than inline, so the
#: whole of what a reader sees can be read — and changed — at once.
#:
#: ``bag`` is kept rather than rendered as *amostra*: CONTEXT.md defines a bag
#: as a sample drawn at a *controlled prevalence* and lists "sample" among the
#: words to avoid for it, precisely because the general word loses that. The
#: phrasing around it is Portuguese; the defined term stays, as defined terms
#: usually do. Change this one entry if the paper prefers *tamanho da
#: amostra*.
LABELS = {
    "title": "Ranking dos métodos por tamanho do bag",
    "subtitle_all": "{n} métodos",
    "subtitle_top": "{shown} melhores de {n} métodos",
    #: Short on purpose: in the stacked figure this is the y-axis label of a
    #: panel only a few inches tall, and "Posição no ranking (1 = melhor)" is
    #: wider than that at this point size — it ran past its own panel and into
    #: the ones above and below.
    "y": "Ranking (1 = melhor)",
    #: Two names for the method axis, because two figures label its ticks
    #: differently: the per-bag-size figures spell the whole method out
    #: (``QuaDapt-Dirichlet(TMS2)``), while the combined ones carry the base
    #: quantifier alone and leave the arm to the colour.
    "x_method": "Método",
    "x_base": "Quantificador base",
    "panel": "tamanho do bag = {bag_size}  ·  {datasets} conjuntos de dados",
    #: The same text over two lines, for the side-by-side figure: there each
    #: panel is a column some seven inches wide and the one-line form is wider
    #: than that, so neighbouring titles ran into each other.
    "panel_stacked": "tamanho do bag = {bag_size}\n{datasets} conjuntos de dados",
}

#: The meta-quantifier arms — what the QuaDapt-only figures draw. Spelled out
#: rather than derived from the colour table, which now also holds the plain
#: arm's grey and so can no longer stand in for "the arms this filters to".
PUBLISHED_METHOD_SIMULATORS = (
    runs.UNIFORM,
    runs.MVN,
    runs.DIRICHLET,
    runs.ALL_SIMULATORS,
)

#: Every arm, meta-quantifier and plain alike — what the all-methods figures
#: draw, in the order their swatches appear in the legend.
ALL_METHOD_SIMULATORS = (*PUBLISHED_METHOD_SIMULATORS, runs.NO_METHOD_SIMULATOR)

#: The bag sizes these figures show, which is not every bag size the table
#: holds (``sweep.BAG_SIZES`` runs to 5000). 5000 is left out because the
#: pools that cannot fill a bag that large drop out of the ranking entirely —
#: it is three datasets of eight on the binary table — so the panel ranks a
#: different, smaller experiment than the ones above it while looking like
#: their continuation. The bag-size sensitivity section is where that belongs;
#: pass ``--bag-size`` to draw it anyway.
PUBLISHED_BAG_SIZES = (100, 500, 1000)

#: Point sizes for a figure that will be shrunk into a column or thrown on a
#: projector. Deliberately much larger than matplotlib's defaults throughout —
#: a tick label is the thing a reader actually needs here, so it is sized like
#: one rather than like an annotation.
FONT_SIZES = {
    "title": 32,
    "axis": 27,
    "tick": 22,
    "legend": 26,
}

OUTPUT_DIR = Path("results")


def quadapt_only(labelled):
    """``labelled``, keeping the meta-quantifier arms and dropping the rest.

    Selected by ``method_simulator`` rather than by matching ``"QuaDapt"`` in
    the method label: the label is built for a reader
    (``runs.method_label``), and filtering on how a name is spelled is exactly
    the defect ``tests/test_renderers.py`` stands against — ``"X"`` and
    ``"MS"`` once quietly filtered away two base quantifiers the sweep records
    as ``"TX"`` and ``"TMS"``.
    """
    return labelled[labelled["method_simulator"].isin(PUBLISHED_METHOD_SIMULATORS)]


def _colours_for(frame, methods):
    """One colour per method, by the simulator of the arm it belongs to."""
    simulator = (
        frame[["method", "method_simulator"]]
        .drop_duplicates()
        .set_index("method")["method_simulator"]
    )
    return [METHOD_SIMULATOR_COLOURS[simulator[method]] for method in methods]


def _base_quantifiers_for(frame, methods):
    """Each method's base quantifier alone — ``TMS2`` for ``QuaDapt-MVN(TMS2)``.

    What a tick label carries once the colour carries the method simulator.
    Spelling both into every tick repeats in text what the legend already says
    in colour, and costs the room that repeating the labels on every panel
    needs — ``QuaDapt-Dirichlet(TMS2)`` is 23 characters where ``TMS2`` is 4.

    Read off the ``base_quantifier`` column rather than parsed back out of the
    label: the label is built for a reader (``runs.method_label``), and
    recovering a field from its own rendering is how a renderer ends up
    filtering on a spelling, which is the defect ``tests/test_renderers.py``
    stands against.
    """
    base = (
        frame[["method", "base_quantifier"]]
        .drop_duplicates()
        .set_index("method")["base_quantifier"]
    )
    return [base[method] for method in methods]


def render(labelled, bag_size, output_dir=OUTPUT_DIR, top=None):
    """The ranking boxplot for one bag size, written as PNG and PDF.

    Returns the paths written, or ``None`` when this bag size has nothing to
    rank — which is a real case, not a guard against one: at bag_size=5000
    neither Haberman nor Pima can fill a bag at all even with #18's
    replication cap.
    """
    frame = quadapt_only(labelled)
    frame = frame[frame["bag_size"] == bag_size]
    if frame.empty:
        print(f"bag_size={bag_size}: no runs, skipped")
        return None

    # The same diagnosis the dashboard and bag_size_sensitivity_report.py
    # print: a dataset with nothing to score here is not a broken method.
    dropped = bag_size_report.datasets_with_no_valid_estimate(frame)
    if dropped:
        print(f"bag_size={bag_size}: no valid estimate for {', '.join(dropped)} — dropped")
        frame = frame[~frame["dataset"].isin(dropped)]

    scored = frame.dropna(subset=["absolute_error"])
    if scored["dataset"].nunique() < 2 or scored["method"].nunique() < 2:
        print(f"bag_size={bag_size}: too few datasets or methods left, skipped")
        return None

    # The same matrix stats.average_ranks reduces to one number per method for
    # the critical-difference diagram; ties take the mean rank.
    ranks = stats.dataset_method_ranks(scored)
    order = ranks.median(axis=0).sort_values().index.tolist()
    n_ranked = len(order)
    if top:
        order = order[:top]

    # Best on the left, reading rank upward on the y-axis. A landscape figure:
    # forty rows stacked down a page does not fit a slide, which is what these
    # are for.
    drawn = order
    colours = _colours_for(scored, drawn)

    fig, ax = plt.subplots(figsize=(max(14.0, 0.58 * len(drawn) + 3.0), 12.5))
    boxes = ax.boxplot(
        [ranks[method].dropna().to_numpy() for method in drawn],
        tick_labels=drawn,
        vert=True,
        patch_artist=True,
        widths=0.68,
        medianprops={"color": "black", "linewidth": 2.6},
        whiskerprops={"linewidth": 1.8},
        capprops={"linewidth": 1.8},
        boxprops={"linewidth": 1.6},
        flierprops={"marker": ".", "markersize": 7, "alpha": 0.45},
    )
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.8)

    shown = (
        LABELS["subtitle_top"].format(shown=len(order), n=n_ranked)
        if top
        else LABELS["subtitle_all"].format(n=n_ranked)
    )
    ax.set_ylabel(LABELS["y"], fontsize=FONT_SIZES["axis"])
    ax.set_xlabel(LABELS["x_method"], fontsize=FONT_SIZES["axis"], labelpad=14)
    ax.set_title(
        LABELS["panel"].format(bag_size=bag_size, datasets=scored["dataset"].nunique())
        + f" — {shown}",
        fontsize=FONT_SIZES["title"],
        # Clears the legend, which sits between the title and the frame. Sized
        # against FONT_SIZES["legend"] rather than left a literal: the two have
        # collided twice now, each time after the fonts were enlarged and this
        # number was not.
        pad=2.8 * FONT_SIZES["legend"],
    )
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"], rotation=90)
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
    ax.grid(axis="y", alpha=0.35, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_SIMULATOR_COLOURS[name], alpha=0.8)
        for name in PUBLISHED_METHOD_SIMULATORS
    ]
    # Above the axes, not inside them: the boxes reach the top of the frame
    # wherever an arm ranks worst on some dataset, so an in-axes corner sits
    # on somebody's whiskers. Above also keeps it clear of the rotated method
    # labels below.
    ax.legend(
        handles,
        # No legend title: "Method simulator" is another line of text between
        # the swatches and the plot title, and the four labels already say
        # what they are.
        [METHOD_SIMULATOR_LABELS[name] for name in PUBLISHED_METHOD_SIMULATORS],
        fontsize=FONT_SIZES["legend"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(handles),
        frameon=False,
        handlelength=1.6,
        columnspacing=2.4,
    )
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"real_data_rank_quadapt_bag{bag_size}"
    png, pdf = output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def _ranks_per_bag_size(labelled, bag_sizes, simulators=None):
    """``(bag_size, ranks, datasets)`` for each bag size with something to rank.

    The ranking is computed separately per bag size and never pooled across
    them (#20): a method at bag 100 and the same method at bag 5000 are two
    measurements of different things, and ranking them together would have a
    method compete against itself. Which is why the combined figure below is
    four panels and not one box per method.
    """
    kept = labelled[
        labelled["method_simulator"].isin(simulators or PUBLISHED_METHOD_SIMULATORS)
    ]
    panels = []
    for bag_size in bag_sizes:
        frame = kept[kept["bag_size"] == bag_size]
        if frame.empty:
            continue
        dropped = bag_size_report.datasets_with_no_valid_estimate(frame)
        if dropped:
            print(f"bag_size={bag_size}: no valid estimate for {', '.join(dropped)} — dropped")
            frame = frame[~frame["dataset"].isin(dropped)]
        scored = frame.dropna(subset=["absolute_error"])
        if scored["dataset"].nunique() < 2 or scored["method"].nunique() < 2:
            print(f"bag_size={bag_size}: too few datasets or methods left, skipped")
            continue
        panels.append((bag_size, stats.dataset_method_ranks(scored), scored))
    return panels


def render_combined(labelled, bag_sizes, output_dir=OUTPUT_DIR, top=None):
    """Every bag size as one figure: stacked panels over a shared method axis.

    Stacked and sharing the x-axis rather than tiled, because the forty method
    names are what costs the space — drawn once under the bottom panel instead
    of once per bag size, the figure is roughly the height of two of the
    separate ones rather than four.

    Every panel is ordered by the same method sequence, so a column is one
    method read across bag sizes and a method that climbs or collapses as bags
    grow is visible as a shape rather than by matching labels between figures.
    The order is each method's mean median-rank over the panels it appears in —
    a neutral choice, since ordering by any single bag size would privilege it.
    """
    panels = _ranks_per_bag_size(labelled, bag_sizes)
    if not panels:
        print("nothing to draw")
        return None

    medians = {}
    for _, ranks, _ in panels:
        for method, median in ranks.median(axis=0).items():
            medians.setdefault(method, []).append(median)
    order = sorted(medians, key=lambda method: np.mean(medians[method]))
    n_ranked = len(order)
    if top:
        order = order[:top]

    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(max(14.0, 0.58 * len(order) + 3.0), 4.6 * len(panels) + 3.0),
    )
    axes = np.atleast_1d(axes)

    for ax, (bag_size, ranks, scored) in zip(axes, panels):
        # A method can be absent from a panel: at bag 5000 the datasets that
        # could not fill a bag are dropped, and a method with no estimate on
        # any survivor has no rank here. An empty list keeps its column in
        # place rather than shifting every method after it by one.
        boxes = ax.boxplot(
            [
                ranks[method].dropna().to_numpy() if method in ranks else []
                for method in order
            ],
            tick_labels=_base_quantifiers_for(scored, order),
            vert=True,
            patch_artist=True,
            widths=0.68,
            medianprops={"color": "black", "linewidth": 2.2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            boxprops={"linewidth": 1.3},
            flierprops={"marker": ".", "markersize": 5, "alpha": 0.4},
        )
        for patch, colour in zip(boxes["boxes"], _colours_for(scored, order)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.8)

        # Every panel gets its own tick labels, not just the bottom one: the
        # panels are a stack of separate rankings, and reading a box in the
        # top panel should not mean tracing a column down two panels to find
        # out which quantifier it is. Affordable only because the label is now
        # the base quantifier alone.
        ax.tick_params(
            axis="x", labelbottom=True, labelsize=FONT_SIZES["tick"], rotation=90
        )
        ax.set_ylabel(LABELS["y"], fontsize=FONT_SIZES["axis"] - 3)
        ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"] - 2)
        ax.grid(axis="y", alpha=0.35, linewidth=1.0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # In-axes, upper left: a label per panel has to sit with its panel, and
        # the top-left corner is where the best-ranked methods leave room.
        ax.text(
            0.006,
            0.95,
            LABELS["panel"].format(
                bag_size=bag_size, datasets=scored["dataset"].nunique()
            ),
            transform=ax.transAxes,
            fontsize=FONT_SIZES["axis"] - 2,
            fontweight="bold",
            va="top",
        )

    # On the bottom panel alone: every panel carries its own tick labels, but
    # an axis *title* repeated three times is three lines of the same words
    # in a figure whose whole point is to save the page.
    axes[-1].set_xlabel(LABELS["x_base"], fontsize=FONT_SIZES["axis"], labelpad=14)

    shown = (
        LABELS["subtitle_top"].format(shown=len(order), n=n_ranked)
        if top
        else LABELS["subtitle_all"].format(n=n_ranked)
    )
    axes[0].set_title(
        f"{LABELS['title']} — {shown}",
        fontsize=FONT_SIZES["title"],
        # Same rule as the per-bag-size figure: derived from the legend's own
        # point size, which is what it has to clear.
        pad=2.8 * FONT_SIZES["legend"],
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_SIMULATOR_COLOURS[name], alpha=0.8)
        for name in PUBLISHED_METHOD_SIMULATORS
    ]
    axes[0].legend(
        handles,
        [METHOD_SIMULATOR_LABELS[name] for name in PUBLISHED_METHOD_SIMULATORS],
        fontsize=FONT_SIZES["legend"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(handles),
        frameon=False,
        handlelength=1.6,
        columnspacing=2.4,
    )
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "real_data_rank_quadapt_all_bag_sizes"
    png, pdf = output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def render_all_methods(labelled, bag_sizes, output_dir=OUTPUT_DIR, top=None):
    """Every method, meta-quantifier and plain, as one vertical figure.

    The counterpart of :func:`render_combined` with nothing filtered out: the
    forty meta-quantifier arms *and* the eleven plain quantifiers, the ``CC``
    baseline among them. Fifty-one of them is too many to run across a page,
    so methods run **down** the y-axis and the bag sizes sit side by side
    sharing it — one column per bag size, read left to right, with the method
    labels once down the left where all three panels touch them.

    That is the opposite arrangement to :func:`render_combined`, and for the
    opposite reason: with forty columns the labels were what cost the height,
    so they were shortened and the panels stacked; with fifty-one rows the
    labels cost nothing to repeat and it is the *boxes* that need the room.

    Tick labels stay the base quantifier alone, as in the combined figure —
    the colour carries the arm, and grey now carries "no meta-quantifier at
    all", so a plain ``DyS`` and a ``QuaDapt-MVN(DyS)`` are one label apart in
    text and a colour apart in fact.
    """
    panels = _ranks_per_bag_size(labelled, bag_sizes, simulators=ALL_METHOD_SIMULATORS)
    if not panels:
        print("nothing to draw")
        return None

    medians = {}
    for _, ranks, _ in panels:
        for method, median in ranks.median(axis=0).items():
            medians.setdefault(method, []).append(median)
    order = sorted(medians, key=lambda method: np.mean(medians[method]))
    n_ranked = len(order)
    if top:
        order = order[:top]

    # Best at the top: matplotlib puts position 1 at the bottom of a
    # horizontal boxplot, so the rows go in worst-first and read upward.
    drawn = order[::-1]

    fig, axes = plt.subplots(
        1,
        len(panels),
        sharey=True,
        figsize=(7.4 * len(panels) + 3.0, max(12.0, 0.42 * len(drawn) + 4.0)),
    )
    axes = np.atleast_1d(axes)

    for ax, (bag_size, ranks, scored) in zip(axes, panels):
        boxes = ax.boxplot(
            [
                ranks[method].dropna().to_numpy() if method in ranks else []
                for method in drawn
            ],
            tick_labels=_base_quantifiers_for(scored, drawn),
            vert=False,
            patch_artist=True,
            widths=0.68,
            medianprops={"color": "black", "linewidth": 2.2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            boxprops={"linewidth": 1.3},
            flierprops={"marker": ".", "markersize": 5, "alpha": 0.4},
        )
        for patch, colour in zip(boxes["boxes"], _colours_for(scored, drawn)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.8)

        ax.set_xlabel(LABELS["y"], fontsize=FONT_SIZES["axis"] - 4)
        ax.set_title(
            LABELS["panel_stacked"].format(
                bag_size=bag_size, datasets=scored["dataset"].nunique()
            ),
            fontsize=FONT_SIZES["axis"] - 4,
            fontweight="bold",
            pad=14,
        )
        ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"] - 3)
        ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"] - 3)
        ax.grid(axis="x", alpha=0.35, linewidth=1.0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # The method axis here is the shared y, so its title belongs to the
    # leftmost panel — the only one carrying the tick labels it names.
    axes[0].set_ylabel(LABELS["x_base"], fontsize=FONT_SIZES["axis"] - 4, labelpad=12)

    shown = (
        LABELS["subtitle_top"].format(shown=len(order), n=n_ranked)
        if top
        else LABELS["subtitle_all"].format(n=n_ranked)
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_SIMULATOR_COLOURS[name], alpha=0.8)
        for name in ALL_METHOD_SIMULATORS
    ]
    fig.legend(
        handles,
        [METHOD_SIMULATOR_LABELS[name] for name in ALL_METHOD_SIMULATORS],
        fontsize=FONT_SIZES["legend"] - 4,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        handlelength=1.6,
        columnspacing=2.0,
        bbox_to_anchor=(0.5, -0.004),
    )
    fig.suptitle(
        f"{LABELS['title']} — {shown}", fontsize=FONT_SIZES["title"], y=0.997
    )
    # Room at the bottom for the legend and at the top for the title, neither
    # of which tight_layout accounts for when they belong to the figure.
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "real_data_rank_todos_metodos_all_bag_sizes"
    png, pdf = output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag-size",
        nargs="+",
        type=int,
        metavar="N",
        help="render only these bag sizes (default: every one in the table)",
    )
    parser.add_argument(
        "--top",
        type=int,
        metavar="N",
        help=(
            "draw only the N best-ranked arms, for a slide that cannot hold "
            "forty columns. The rank itself is always computed over every arm, "
            "so a cut figure still reports where a method placed among all of "
            "them rather than among the few drawn"
        ),
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help=(
            "draw every method, the eleven plain quantifiers and the CC "
            "baseline included, instead of the meta-quantifier arms alone. "
            "Fifty-one does not fit across a page, so this figure runs the "
            "methods down the y-axis with the bag sizes side by side, and "
            "writes under its own name rather than over the arms-only one"
        ),
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help=(
            "write one figure per bag size instead of the single combined one "
            "(the default), which stacks every bag size over a shared method "
            "axis and so costs about half the page the separate ones do"
        ),
    )
    args = parser.parse_args()

    labelled = runs.load_labelled(runs.REAL_DATA)
    present = {int(size) for size in labelled["bag_size"].unique()}
    bag_sizes = args.bag_size or [b for b in PUBLISHED_BAG_SIZES if b in present]

    # Said rather than silently dropped: a figure that covers less than the
    # table does should say so where the figure is made, not only in a comment.
    skipped = sorted(present - set(bag_sizes))
    if skipped:
        print(f"bag sizes in the table but not drawn: {skipped} (pass --bag-size to include)")

    print(
        f"{len(labelled)} runs, {labelled['dataset'].nunique()} datasets; "
        f"{quadapt_only(labelled)['method'].nunique()} meta-quantifier methods "
        f"of {labelled['method'].nunique()}"
    )
    if args.separate:
        for bag_size in bag_sizes:
            written = render(labelled, bag_size, top=args.top)
            if written:
                print(" and ".join(str(path) for path in written))
        return

    render_figure = render_all_methods if args.all_methods else render_combined
    written = render_figure(labelled, bag_sizes, top=args.top)
    if written:
        print(" and ".join(str(path) for path in written))


if __name__ == "__main__":
    _main()
