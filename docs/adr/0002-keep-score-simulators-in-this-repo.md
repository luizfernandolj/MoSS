# Keep the score simulators here rather than upstreaming them to mlquantify

The MVN and Dirichlet simulators are general enough to belong in mlquantify —
which this project's author maintains — and they are written to satisfy the
`MoSS(n, alpha, merging_factor, classes, random_state)` seam mlquantify already
defines. We nonetheless keep them in this repository until the study is
published, so that results depend on a released library version rather than one
co-evolving with the experiments that cite it.

Matching the upstream signature is deliberate: it keeps upstreaming a file move
rather than a rewrite, once there is no longer a paper pinned to the behaviour.
