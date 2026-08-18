"""Preregistered statistical analysis for the MAPG benchmark (item 9).

Offline analysis only. Numpy is allowed here, unlike the runner-side
modules, because this package never runs inside the Habitat container.

Modules:
- endpoints:   endpoint definitions, censoring, per-query aggregation,
               row-count integrity, the preregistered K = 5 family
- bootstrap:   hierarchical (scene-cluster) BCa bootstrap CIs
- permutation: cluster permutation test and Holm-Bonferroni
- report:      python3 -m src.stats.report --db <path> --split <name>
"""
