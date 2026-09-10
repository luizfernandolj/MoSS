# MoSS

Experimental study of score simulation for quantification: whether simulating a
classifier's score distribution gives a quantifier a better reference than the
real training scores, and which family of simulator does it best.

## Language

### Score simulation

**Score simulator**:
A generator that produces soft classification scores with known labels from a
sample count, a prevalence and a merging factor.
_Avoid_: MoSS variant, generator, distribution

**Uniform simulator**:
The original MoSS score simulator, drawing positive scores as `U^m` and negative
scores as `1 - U^m`. Binary only.
_Avoid_: MoSS (bare, when a specific simulator is meant)

**MVN simulator**:
A score simulator placing class centroids on the simplex vertices and drawing
from a diagonal multivariate normal around them. Generalises to any class count.
_Avoid_: MoSS_MN, normal, gaussian

**Dirichlet simulator**:
A score simulator drawing directly from a Dirichlet whose concentration is
derived from the merging factor. Generalises to any class count.
_Avoid_: MoSS_Dir

**Merging factor**:
How much the positive and negative score distributions overlap: 0 is fully
separated, 1 is fully merged. The single knob that sets a simulated score set's
difficulty.
_Avoid_: m, dispersion, variance, overlap, complexity

### The two roles a simulator plays

These are the same generators used for entirely different purposes. Naming them
apart is the point: conflating them is what made a vacuous experimental axis
invisible for 3.75M runs.

**Data simulator**:
A score simulator in its role as the source of an experiment's own training or
test scores. Exists only in the synthetic experiment — on real data the
classifier and the dataset supply these.
_Avoid_: MoSS variant, train variant, test variant

**Method simulator**:
A score simulator in its role inside a meta-quantifier, generating the candidate
score sets it chooses between. A property of the estimator, so it is present in
both the synthetic and the real-data experiment.
_Avoid_: QuaDapt variant, Quadapt_Variant

### Quantification

**Prevalence**:
The proportion of each class in a sample. The quantity being estimated.
_Avoid_: alpha (overloaded — also names a Dirichlet concentration), class ratio,
class distribution

**Bag**:
A sample drawn at a controlled prevalence. The unit a quantifier produces one
estimate for.
_Avoid_: batch, test set, sample

**Protocol**:
The rule for drawing a series of bags at controlled prevalences from a fixed
pool, so that estimation error can be measured across the prevalence range.

**Quantifier**:
A method that estimates a bag's prevalence directly, rather than by classifying
its members and counting.

**Meta-quantifier**:
A quantifier that does not estimate. It selects a reference score set and
delegates the estimate to a base quantifier.
_Avoid_: wrapper, ensemble

**Distance measure**:
What a meta-quantifier's mixture search minimises when it scores a candidate
score set against a bag. Fixed for the published synthetic experiment
(ADR-0012); which measure estimates best is the measure-ablation experiment's
question, not the published grid's.
_Avoid_: metric, distance function

**Base quantifier**:
The quantifier a meta-quantifier wraps and delegates its estimate to.
_Avoid_: inner quantifier, learner, estimator

**Method**:
A base quantifier together with the method simulator of the meta-quantifier
wrapping it, if any. The unit a run is attributed to and a figure draws one
line for: `DyS` and `QuaDapt-MVN(DyS)` are two methods, not one.
_Avoid_: variant, approach, algorithm

**Reference score set**:
The labelled score distribution a quantifier matches a bag against. Either the
real training scores or a simulated substitute.
_Avoid_: train scores (ambiguous once the reference may be simulated)

**Candidate score set**:
One reference score set among the several a meta-quantifier evaluates before
picking one. Usually simulated, but not necessarily: the meta-quantifier that
searches every simulator's candidates counts the real reference scores among
them, so that "none of the substitutes beat the real thing" is an answer it
can give (ADR-0010).

### Experiments

**Run**:
One estimate: one bag, one method, one base quantifier, with the true and
estimated prevalence recorded together.
_Avoid_: result, row, experiment

**Sweep**:
The full cross-product of runs making up one experiment.

**Synthetic experiment**:
The sweep where both the reference and the bags come from data simulators, so
that score-distribution difficulty is a controlled variable.

**Real-data experiment**:
The sweep where the reference comes from a classifier's cross-validated scores
and bags are drawn from a real dataset under a protocol.

**Measure-ablation experiment**:
The sweep run once per distance measure, on a grid smaller than the synthetic
experiment's, asking which measure a meta-quantifier's mixture search should
minimise. Its own table (ADR-0012), because the measure it varies is fixed for
every other experiment.
