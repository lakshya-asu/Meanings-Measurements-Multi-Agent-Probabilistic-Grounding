"""Object-selection scoring for O-O and MCQ (item 6 of the harness).

Implements the preregistered design from research/metrics.md section 2.3:

Primary rule: exact node-ID matching. A frozen GT_object -> Hydra node-ID
mapping ships with the benchmark (see src/scripts/build_node_id_map.py).
Systems emit {"node_id": <int>}; strict accuracy is exact ID equality.
Deterministic, no judge variance.

Lenient cascade, for systems that emit only free text. Applied in order,
first hit wins:
  1. Canonicalize: NFKC normalize, lowercase, strip articles and
     punctuation, singularize (naive, with common irregulars), map
     through the frozen synonym table splits/synonyms_v1.json.
  2. Exact string match against canonicalized node labels in the same
     scene.
  3. Embedding fallback (sentence-transformers/all-mpnet-base-v2 at a
     pinned commit, mean pooled, L2 normalized), cosine >= 0.75. The
     embedding step is PLUGGABLE here: pass an ``embedder(texts) ->
     list[vector]`` callable. When None (host env without
     sentence-transformers) the result records ``embedding_unavailable``
     and the query is unmatched unless steps 1-2 hit.
  4. Deterministic tie-breaks, in order: higher cosine, smaller centroid
     distance to the GT node, lower node ID.
  5. Nothing at or above 0.75 means unmatched, counted INCORRECT, never
     dropped.

Top-1 everywhere. No best-of-Top-k.

Hooks left for later columns (out of scope today): distractor-conditioned
accuracy and chance-adjusted MCQ accuracy will be computed downstream from
per-query MatchResult rows plus scene metadata; MatchResult carries enough
(step, cosine, tie_break) that no rescoring will be needed.

Stdlib only (json, math, string, unicodedata, dataclasses) so this module
imports on the host and inside the Habitat container alike.
"""

from __future__ import annotations

import json
import math
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

# Repo root is two levels up from this file (src/evals/object_selection.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SYNONYMS_PATH = _REPO_ROOT / "splits" / "synonyms_v1.json"

# Cosine threshold for the embedding fallback (preregistered).
EMBED_THRESHOLD = 0.75

# Articles stripped during canonicalization.
_ARTICLES = {"a", "an", "the"}

# Common irregular plurals seen in indoor-scene labels. The general rule
# below handles regular -s / -es / -ies forms.
_IRREGULAR_SINGULARS = {
    "shelves": "shelf",
    "knives": "knife",
    "leaves": "leaf",
    "children": "child",
    "men": "man",
    "women": "woman",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
}

_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def load_synonyms(path: Optional[Path] = None) -> dict:
    """Load the frozen synonym table, dropping the _meta block."""
    p = Path(path) if path is not None else SYNONYMS_PATH
    with open(p, encoding="utf-8") as f:
        table = json.load(f)
    return {k: v for k, v in table.items() if k != "_meta"}


def _singularize_word(word: str) -> str:
    if word in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[word]
    # Words that end in these are not plural forms (glass, cactus, iris).
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2].endswith(("sh", "ch", "x", "z", "s")):
        return word[:-2]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def canonicalize(label: str, synonyms: Optional[dict] = None) -> str:
    """Canonicalize a free-text object label (cascade step 1).

    NFKC normalize, lowercase, strip articles and punctuation, collapse
    whitespace, naively singularize each word, then map the whole phrase
    through the synonym table. Deterministic.
    """
    if synonyms is None:
        synonyms = load_synonyms()
    text = unicodedata.normalize("NFKC", label).lower()
    text = text.translate(_PUNCT_TABLE)
    words = [w for w in text.split() if w not in _ARTICLES]
    words = [_singularize_word(w) for w in words]
    phrase = " ".join(words)
    return synonyms.get(phrase, phrase)


def strict_accuracy(pred_node_id, gt_node_id) -> bool:
    """Strict rule: exact node-ID match (primary metric).

    Returns True iff the predicted node ID is an int exactly equal to the
    GT node ID. Anything else (None, wrong type, wrong ID) is False.
    Aggregate strict accuracy is the mean of this over all queries.
    """
    if isinstance(pred_node_id, bool) or not isinstance(pred_node_id, int):
        return False
    if isinstance(gt_node_id, bool) or not isinstance(gt_node_id, int):
        raise ValueError(f"GT node id must be an int, got {gt_node_id!r}")
    return pred_node_id == gt_node_id


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two plain vectors, stdlib math only."""
    if len(u) != len(v):
        raise ValueError(f"vector length mismatch: {len(u)} vs {len(v)}")
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of the lenient cascade for one query.

    step is which cascade stage decided the result:
      "exact_string"          step 2 hit
      "embedding"             step 3 hit (cosine >= threshold)
      "embedding_unavailable" steps 1-2 missed and no embedder was given
      "unmatched"             embedder ran, nothing reached the threshold
    Unmatched and embedding_unavailable are counted INCORRECT, never
    dropped (correct is False, matched is False).

    tie_break records the deterministic tie-break path taken, e.g.
    "cosine", "cosine>centroid", "centroid>node_id"; None when a single
    candidate won outright.
    """
    matched: bool
    correct: bool
    node_id: Optional[int]
    step: str
    cosine: Optional[float]
    tie_break: Optional[str]
    pred_canonical: str


def _centroid_distance(node: dict, gt_centroid) -> float:
    c = node.get("centroid")
    if c is None or gt_centroid is None:
        # No geometry available: neutral value, tie falls to node ID.
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c, gt_centroid)))


def _select(candidates, gt_centroid):
    """Pick one winner from (node, cosine) pairs, deterministically.

    Order: higher cosine, then smaller centroid distance to the GT node,
    then lower node ID. Returns (node, cosine, tie_break_path).
    """
    keyed = sorted(
        candidates,
        key=lambda nc: (
            -nc[1],
            _centroid_distance(nc[0], gt_centroid),
            int(nc[0]["node_id"]),
        ),
    )
    best, best_cos = keyed[0]
    if len(keyed) == 1:
        return best, best_cos, None

    # Reconstruct which comparisons the winner needed against the
    # runner-up, so the path is auditable.
    runner, runner_cos = keyed[1]
    path = []
    if best_cos != runner_cos:
        path.append("cosine")
    else:
        path.append("cosine")  # compared, tied
        d_best = _centroid_distance(best, gt_centroid)
        d_run = _centroid_distance(runner, gt_centroid)
        if d_best != d_run:
            path.append("centroid")
        else:
            path.append("centroid")  # compared, tied
            path.append("node_id")
    # Drop the leading comparisons that were ties only if the decisive
    # key is the first one; keep the full path otherwise.
    if len(path) == 1:
        return best, best_cos, "cosine"
    return best, best_cos, ">".join(path)


def lenient_match(
    pred_text: str,
    scene_nodes: list,
    gt_node_id: int,
    embedder: Optional[Callable] = None,
    synonyms: Optional[dict] = None,
    threshold: float = EMBED_THRESHOLD,
) -> MatchResult:
    """Run the lenient cascade for a free-text prediction (Top-1 only).

    scene_nodes: node dicts for the SAME scene, each with "node_id" (int),
    "label" (str), and optionally "centroid" ([x, y, z]).
    embedder: optional callable, embedder(list[str]) -> list[vector].
    Vectors are plain sequences of floats; cosine is computed here with
    stdlib math. When embedder is None the embedding step is skipped and
    recorded as embedding_unavailable.
    """
    if synonyms is None:
        synonyms = load_synonyms()

    pred_canon = canonicalize(pred_text, synonyms)
    canon_nodes = [
        (node, canonicalize(node["label"], synonyms)) for node in scene_nodes
    ]
    gt_centroid = None
    for node in scene_nodes:
        if int(node["node_id"]) == int(gt_node_id):
            gt_centroid = node.get("centroid")
            break

    # Step 2: exact string match on canonical labels in the same scene.
    exact_hits = [node for node, canon in canon_nodes if canon == pred_canon]
    if exact_hits:
        # All exact hits are equally good on the string key, so the
        # cosine slot is a constant 1.0 and ties fall through to the
        # geometric and ID tie-breaks.
        winner, _, tie = _select([(n, 1.0) for n in exact_hits], gt_centroid)
        # The cosine slot was a constant here, so it is not a real leg of
        # the tie-break path at this step.
        if tie is not None and tie.startswith("cosine>"):
            tie = tie[len("cosine>"):]
        node_id = int(winner["node_id"])
        return MatchResult(
            matched=True,
            correct=(node_id == int(gt_node_id)),
            node_id=node_id,
            step="exact_string",
            cosine=None,
            tie_break=tie,
            pred_canonical=pred_canon,
        )

    # Step 3: embedding fallback, pluggable.
    if embedder is None:
        return MatchResult(
            matched=False,
            correct=False,
            node_id=None,
            step="embedding_unavailable",
            cosine=None,
            tie_break=None,
            pred_canonical=pred_canon,
        )

    texts = [pred_canon] + [canon for _, canon in canon_nodes]
    vectors = embedder(texts)
    pred_vec = vectors[0]
    scored = [
        (node, cosine_similarity(pred_vec, vec))
        for (node, _), vec in zip(canon_nodes, vectors[1:])
    ]
    best_seen = max((c for _, c in scored), default=None)
    candidates = [(node, c) for node, c in scored if c >= threshold]
    if not candidates:
        # Step 5: below threshold everywhere. Unmatched, counted
        # incorrect, never dropped.
        return MatchResult(
            matched=False,
            correct=False,
            node_id=None,
            step="unmatched",
            cosine=best_seen,
            tie_break=None,
            pred_canonical=pred_canon,
        )

    winner, win_cos, tie = _select(candidates, gt_centroid)
    node_id = int(winner["node_id"])
    return MatchResult(
        matched=True,
        correct=(node_id == int(gt_node_id)),
        node_id=node_id,
        step="embedding",
        cosine=win_cos,
        tie_break=tie,
        pred_canonical=pred_canon,
    )
