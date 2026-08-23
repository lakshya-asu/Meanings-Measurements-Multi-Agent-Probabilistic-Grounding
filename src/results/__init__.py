"""Gate 4 of the research harness: SQLite results store + run manifest.

One row per episode in a WAL-mode SQLite database, plus a run.json
manifest that pins git SHA, config hash, seed and split SHA for every
run. Legacy JSON logging stays in place next to this; nothing here
replaces it yet.
"""

from src.results.store import ResultsStore
from src.results.manifest import build_manifest, write_manifest

__all__ = ["ResultsStore", "build_manifest", "write_manifest"]
