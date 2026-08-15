"""MAPG-01: kernel bandwidth sizing from real Hydra AABB extents.

Pure helpers (stdlib only) shared by MultiAgentMSPPlanner and the host
test suite. The MSP engine (MSPEngineSmart._build_anchor_only_params)
derives every bandwidth from the max entry of the anchor "size" it is
handed:

    sigma_s = sigma_s_factor * max(size)            (factor default 0.5)
    sigma_m = max(sigma_m_factor * max(size), sigma_m_floor)
    kappa   = max(kappa_factor / max(max(size), 0.1), 0.5)

Under cfg ``kernel_bandwidths: from_bbox`` (the default) the planner
hands the engine an isotropic cube [s, s, s] where

    s = max horizontal extent of the anchor AABB
      = max(extent_x, extent_y) in the z-up Hydra world frame
      = max(size_hab[0], size_hab[2]) in the habitat y-up frame

so the engine's max(size) is exactly s and

    sigma_s = sigma_s_factor * max_horizontal_extent_m.

The vertical extent is excluded on purpose: the density is evaluated
planar (planar=True everywhere in the multi-agent path), and a tall
thin object (floor lamp, fridge) should not widen its horizontal
kernels just because it is tall.

Under ``kernel_bandwidths: fixed`` (the ablation arm) the planner hands
the engine the historical 0.5 m cube, byte-for-byte reproducing the
pre-MAPG-01 behavior (sigma_s = 0.25 m for every anchor). A node whose
box is missing or invalid always falls back to the fixed cube with a
recorded reason; it never crashes the step.
"""

import math

# The historical hardcoded candidate size (pre-MAPG-01 behavior).
FIXED_SIZE_M = (0.5, 0.5, 0.5)

MODES = ("fixed", "from_bbox")
DEFAULT_MODE = "from_bbox"


def resolve_mode(raw):
    """Normalize the cfg value for ``kernel_bandwidths``.

    Returns (mode, warning). Unknown or missing values fall back to
    DEFAULT_MODE; the warning is None unless the value was unknown.
    """
    if raw is None:
        return DEFAULT_MODE, None
    mode = str(raw).lower().strip()
    if mode not in MODES:
        return DEFAULT_MODE, (
            f"unknown kernel_bandwidths={raw!r}; using {DEFAULT_MODE!r}"
        )
    return mode, None


def extents_zup_to_size_hab(extents_zup):
    """z-up extents [dx, dy, dz_up] -> habitat y-up size [dx, dz_up, dy].

    Matches pos_normal_to_habitat (+90 deg about x): habitat y is the
    z-up vertical, habitat x/z are the horizontal axes. Extents are
    unsigned so no sign flip is needed.
    """
    dx, dy, dz = (float(v) for v in list(extents_zup)[:3])
    return [dx, dz, dy]


def resolve_object_size_hab(extents_zup, mode):
    """Resolve one candidate's LLM-visible size. Never raises.

    Returns (size_hab, source, reason):
      source 'hydra_bbox'     real extents, habitat y-up order;
      source 'fixed_cfg'      kernel_bandwidths == 'fixed';
      source 'fixed_fallback' box missing/invalid, reason says why.
    """
    if mode == "fixed":
        return list(FIXED_SIZE_M), "fixed_cfg", None
    try:
        if extents_zup is None:
            return list(FIXED_SIZE_M), "fixed_fallback", "node has no bbox extents"
        ext = [float(v) for v in list(extents_zup)[:3]]
        if len(ext) != 3:
            return (
                list(FIXED_SIZE_M),
                "fixed_fallback",
                f"extents wrong arity: {extents_zup!r}",
            )
        if any(not math.isfinite(v) or v <= 0.0 for v in ext):
            return (
                list(FIXED_SIZE_M),
                "fixed_fallback",
                f"extents not finite-positive: {ext}",
            )
        return extents_zup_to_size_hab(ext), "hydra_bbox", None
    except Exception as e:
        return list(FIXED_SIZE_M), "fixed_fallback", f"extents unreadable: {e}"


def max_horizontal_extent_hab(size_hab):
    """Max horizontal extent (m) of a habitat y-up size: max(x, z)."""
    return float(max(float(size_hab[0]), float(size_hab[2])))


def bandwidth_size_hab(size_hab, source):
    """The isotropic size actually handed to the MSP engine.

    Returns (cube, scale_m). For source 'hydra_bbox' the cube is
    [s, s, s] with s = max_horizontal_extent_hab(size_hab), so the
    engine derives sigma_s = sigma_s_factor * s. Any other source
    (fixed_cfg, fixed_fallback) yields the historical 0.5 m cube.
    """
    if source == "hydra_bbox":
        s = max_horizontal_extent_hab(size_hab)
        return [s, s, s], s
    return list(FIXED_SIZE_M), float(max(FIXED_SIZE_M))


def sigma_s_m(scale_m, sigma_s_factor=0.5):
    """Mirror of the engine's sigma_s derivation for tests and docs:
    sigma_s = sigma_s_factor * scale, where scale = max(engine size)."""
    return float(sigma_s_factor) * float(scale_m)
