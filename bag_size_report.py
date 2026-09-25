"""How bag size affects estimation error, shaped for a figure rather than eyeballed from a raw table (#19).

Two questions, two functions, the same split #19's own title draws between
"MAE trend" and "per-bag-size rankings". :func:`mae_by_bag_size` answers "does
error trend down as bags grow" at the altitude a reader compares datasets at —
one number per (dataset, bag_size), methods and repetitions averaged away.
:func:`rankings_by_bag_size` answers the question that average cannot: "does
*which method wins* change as bags grow" — computed once per bag size rather
than once across all of them, because a ranking that mixed every bag size
together would average a shift away exactly where a reader most wants to see
one.

Both take ``labelled_runs`` the way ``runs.load_labelled`` returns them, with
``bag_size`` alongside the ``dataset``, ``method`` and ``absolute_error``
columns ``stats.py`` already reads. Neither filters on replication: a bag a
small pool could only fill by bootstrap-replicating a short class (#18) is
still a real bag drawn at that bag size, and dropping it would silently thin
the sample a bag-size trend depends on most at the sizes where pools run out —
exactly the rows this reporting exists to show.

Drawing either shape is not this module's job, the same split ``grid.py`` and
``export_grid_matplotlib.py`` draw for the synthetic figure: this module only
shapes, ``bag_size_sensitivity_report.py`` only draws.
"""

from dataclasses import dataclass

import stats

#: The columns a bag-size trend groups on. Everything else a run carries
#: (method, repetition, target prevalence, ...) collapses into the mean —
#: methods on purpose, since separating them is :func:`rankings_by_bag_size`'s
#: job, not this one's.
_MAE_GROUP_COLUMNS = ("dataset", "bag_size")

#: The column :func:`rankings_by_bag_size` splits its ranking on.
_BAG_SIZE_COLUMN = "bag_size"


def mae_by_bag_size(labelled_runs):
    """Mean absolute error per (dataset, bag_size): one point per line a trend figure draws.

    Averaging over methods here, rather than leaving them apart, is the
    difference between this figure and the critical-difference diagrams
    :func:`rankings_by_bag_size` feeds: this one asks whether bags growing
    changes error at all, not which method that error belongs to.
    """
    return (
        labelled_runs.groupby(list(_MAE_GROUP_COLUMNS), observed=True)["absolute_error"]
        .mean()
        .reset_index()
        .sort_values(list(_MAE_GROUP_COLUMNS))
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class BagSizeRanking:
    """One bag size's own Friedman/Nemenyi ranking, computed independently of every other bag size's.

    Carrying both rather than ``nemenyi`` alone: ``friedman`` is what licenses
    reading anything into ``nemenyi`` at all (Demsar, 2006, and ``stats.py``'s
    own docstring), and a reader asking whether bag size shifts a ranking
    needs that license restated at every bag size, not only at the one where
    the whole table was pooled.

    ``dropped_datasets`` names whichever datasets this bag size's ranking
    left out (empty for the ordinary case) — see
    :func:`datasets_with_no_valid_estimate` for why a dataset is ever
    dropped rather than every ranking failing outright.
    """

    bag_size: int
    friedman: stats.FriedmanResult
    nemenyi: stats.NemenyiResult
    dropped_datasets: tuple = ()


def datasets_with_no_valid_estimate(group):
    """Which of ``group``'s datasets have not one valid estimate to average (#20).

    Distinct from a dataset that is merely missing *some* estimates (#18's
    ordinary case, left to average over whatever it has): the published run
    surfaced datasets whose pool is too small to fill a bag at all at
    bag_size=5000, even with #18's bootstrap-replication cap — every method's
    estimate is missing for that dataset, every repetition, every prevalence.
    ``stats.friedman``'s per-method check would call that "method X has no
    valid estimate", which is the wrong diagnosis: nothing is wrong with the
    method, the dataset itself has nothing to score at this bag size.

    A dataset counts as valid the moment *any* method has an estimate on it,
    rather than checking each method separately: ``real_data.run_cell`` draws
    one bag per repetition and shares it across every method, so a bag-draw
    failure blanks every method's row for that cell at once, and there is no
    path today to one method going dark on a dataset while its others stay
    scored. A future failure mode that broke one method alone across a whole
    dataset would slip past this check and still raise inside ``stats.py`` —
    correctly, since that would be an actual method defect rather than a
    dataset with nothing to score.
    """
    with_estimate = set(group.dropna(subset=["absolute_error"])["dataset"].unique())
    return tuple(sorted(set(group["dataset"].unique()) - with_estimate))


def rankings_by_bag_size(labelled_runs, alpha=0.05):
    """Friedman and Nemenyi computed separately within each bag size, ordered by bag size ascending.

    Grouping on ``bag_size`` before calling :func:`stats.friedman` and
    :func:`stats.nemenyi` is the whole function: a ranking computed across
    every bag size at once would treat two runs of the same method at
    bag_size 100 and bag_size 5000 as two independent observations of the same
    thing, which is exactly the shift this report exists to detect rather than
    average away.

    A dataset with nothing to contribute at this bag size
    (:func:`datasets_with_no_valid_estimate`) is dropped before either test
    runs, rather than failing the whole bag size's ranking over the one
    dataset that has nothing to say at that size.
    """
    rankings = []
    for bag_size, group in labelled_runs.groupby(_BAG_SIZE_COLUMN, observed=True):
        dropped = datasets_with_no_valid_estimate(group)
        if dropped:
            group = group[~group["dataset"].isin(dropped)]
        rankings.append(
            BagSizeRanking(
                bag_size=int(bag_size),
                friedman=stats.friedman(group),
                nemenyi=stats.nemenyi(group, alpha=alpha),
                dropped_datasets=dropped,
            )
        )
    return tuple(rankings)
