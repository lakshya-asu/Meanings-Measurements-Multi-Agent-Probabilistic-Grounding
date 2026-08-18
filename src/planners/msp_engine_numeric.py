"""Numeric-conditioning-aware MSP engine (P0 fix 1: d0 single source).

Subclasses MSPEngineSmart (the anchor-only engine the multi-agent
planner uses) so that ``question_dist`` / ``distance_m`` may be None.

Composition change for no-distance queries
------------------------------------------
None means the utterance carries no metric literal. In that case the
metric (radial) expert is OMITTED from the product-of-experts
composition instead of being fabricated at 1.0 m (the old
``_parse_q_dist`` silent default). Because src/msp/pdf.py has a fixed
params contract (d0 and sigma_m are always read), omission is
implemented as d0 = 0.0 with sigma_m = 1e6: the radial exponent is then
numerically zero for any indoor-scale distance, exactly the mechanism
the engine already uses to delete the semantic kernel
(flatten_semantic -> sigma_s = 1e6). The debug record carries
``metric_kernel_active`` so ablations can condition on it.

Point estimation without a metric kernel: the radial proposal cannot be
a Gaussian around d0 anymore, so radii are drawn uniformly from a fixed
indoor range (0.2 to 5.0 m). That range is a PROPOSAL support, not a
kernel parameter: the density still contains no radial preference, and
the chosen point's radius is decided by the remaining experts (and is
otherwise unconstrained, which is the honest reading of a query with no
stated distance).

GT front yaw helper (P0 fix 2 plumbing): ``gt_front_theta`` maps the
bench annotation ann_yaw_rad (anchor front yaw, world radians, y-up
convention: yaw in the x-z plane from +x toward +z) plus an intrinsic
relation label to the world yaw of the target direction. It is only
called when cfg ``frames.gt_front`` is true (oracle-frames ablation);
the default pipeline records the annotation but never uses it.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.planners.vlm_planner_msp_debug import MSPEngineSmart
from src.msp.pdf import combined_logpdf as _combined_logpdf

# Flat sigma for a dropped kernel, matching the flatten_semantic idiom.
FLAT_SIGMA_M = 1.0e6

# Radial proposal support (meters) when the metric kernel is omitted.
NO_METRIC_RADIAL_PROPOSAL_RANGE_M = (0.2, 5.0)


def gt_front_theta(front_yaw_rad: float, relation: Optional[str]) -> Optional[float]:
    """World yaw of the target direction from the GT anchor front yaw.

    Only intrinsic relations are mapped; anything else returns None (no
    override). Offsets assume yaw measured from +x toward +z (y-up):
    left = front + pi/2, right = front - pi/2. That sign convention is
    an assumption recorded here in one place; the oracle-frames
    ablation should sanity-check it against src/evals/angular.py.
    """
    rel = (relation or "").strip().lower().replace(" ", "_")
    offsets = {
        "intrinsic_front": 0.0,
        "in_front_of": 0.0,
        "intrinsic_back": math.pi,
        "behind": math.pi,
        "intrinsic_left": math.pi / 2.0,
        "left_of": math.pi / 2.0,
        "intrinsic_right": -math.pi / 2.0,
        "right_of": -math.pi / 2.0,
    }
    if rel not in offsets:
        return None
    two_pi = 2.0 * math.pi
    return (float(front_yaw_rad) + offsets[rel]) % two_pi


class NumericConditioningEngine(MSPEngineSmart):
    """MSPEngineSmart that accepts question_dist=None (metric kernel off)."""

    def _build_anchor_only_params(
        self,
        anchor_pos_hab: np.ndarray,
        anchor_size: Optional[List[float]],
        distance_m: Optional[float],
        kernel_params: Dict[str, Any],
        planar: bool = True,
        flatten_semantic: bool = False,
    ) -> Dict[str, float]:
        metric_active = distance_m is not None
        params = super()._build_anchor_only_params(
            anchor_pos_hab=anchor_pos_hab,
            anchor_size=anchor_size,
            distance_m=(float(distance_m) if metric_active else 0.0),
            kernel_params=kernel_params,
            planar=planar,
            flatten_semantic=flatten_semantic,
        )
        if not metric_active:
            # Omit the metric expert: flat radial Gaussian, no peak.
            params["d0"] = 0.0
            params["sigma_m"] = FLAT_SIGMA_M
        # Underscore prefix: stripped before the pdf call, kept in debug.
        params["_metric_kernel_active"] = bool(metric_active)
        return params

    def _debug_record(
        self,
        anchor_pos_hab: np.ndarray,
        anchor_size: Optional[List[float]],
        pos_hab: np.ndarray,
        question_dist: Optional[float],
        kernel_params: Dict[str, Any],
        logp: float,
        planar: bool = True,
        flatten_semantic: bool = False,
    ) -> Dict[str, Any]:
        # Same record as the base class, but question_dist may be None
        # and metric_kernel_active is recorded for ablations.
        pos = np.asarray(pos_hab, dtype=np.float32)
        params = self._build_anchor_only_params(
            anchor_pos_hab=np.asarray(anchor_pos_hab, dtype=np.float32),
            anchor_size=anchor_size,
            distance_m=question_dist,
            kernel_params=kernel_params,
            planar=planar,
            flatten_semantic=flatten_semantic,
        )
        rec = {
            "candidate_pos_hab": [float(pos[0]), float(pos[1]), float(pos[2])],
            "question_dist_m": (float(question_dist) if question_dist is not None else None),
            "anchor_size": [float(x) for x in (anchor_size or [0.5, 0.5, 0.5])],
            "anchor_max_dim": float(params.get("_anchor_max_dim", 0.0)),
            "factors": {
                "sigma_s_factor": float(self.sigma_s_factor),
                "sigma_m_factor": float(self.sigma_m_factor),
                "kappa_factor": float(self.kappa_factor),
                "sigma_m_floor": float(self.sigma_m_floor),
            },
            "metric_semantic_params": {
                "mu_x": float(params["mu_x"]),
                "mu_y": float(params["mu_y"]),
                "mu_z": float(params["mu_z"]),
                "sigma_s": float(params["sigma_s"]),
                "x0": float(params["x0"]),
                "y0": float(params["y0"]),
                "z0": float(params["z0"]),
                "d0": float(params["d0"]),
                "sigma_m": float(params["sigma_m"]),
                "metric_kernel_active": bool(params.get("_metric_kernel_active", True)),
            },
            "predicate_params": {
                "theta0": float(params["theta0"]),
                "phi0": float(params["phi0"]),
                "kappa": float(params["kappa"]),
                # keep VLM kappa for audit (not used for scoring)
                "vlm_kappa_raw": float(kernel_params.get("kappa", 0.0)),
            },
            "combined_logpdf": float(logp),
            "flatten_semantic": bool(flatten_semantic),
        }
        return rec

    def estimate_point_from_pdf(
        self,
        anchor_pos_hab: np.ndarray,
        kernel_params: Dict[str, Any],
        question_dist: Optional[float],
        anchor_size: Optional[List[float]] = None,
        num_samples: int = 2048,
        planar: bool = True,
        use_map: bool = False,
        seed: int = 0,
        flatten_semantic_for_where: bool = True,
    ) -> Dict[str, Any]:
        # Same estimator as the base class except the radial proposal:
        # Gaussian around d0 when a metric literal exists, uniform over
        # NO_METRIC_RADIAL_PROPOSAL_RANGE_M when the metric kernel is
        # omitted (see module docstring).
        rng = np.random.default_rng(int(seed))
        anchor_pos_hab = np.asarray(anchor_pos_hab, dtype=np.float32)

        params = self._build_anchor_only_params(
            anchor_pos_hab=anchor_pos_hab,
            anchor_size=anchor_size,
            distance_m=question_dist,
            kernel_params=kernel_params,
            planar=planar,
            flatten_semantic=bool(flatten_semantic_for_where),
        )
        metric_active = bool(params.get("_metric_kernel_active", True))

        sigma_m = float(params["sigma_m"])
        theta0 = float(params["theta0"])
        phi0 = float(params["phi0"])
        kappa = float(params["kappa"])

        theta_std = float(1.0 / max(np.sqrt(max(kappa, 1e-6)), 1e-6))
        theta_samples = theta0 + rng.normal(0.0, theta_std, size=(num_samples,)).astype(np.float32)

        if planar:
            phi_samples = np.full((num_samples,), np.float32(phi0), dtype=np.float32)
        else:
            phi_std = theta_std
            phi_samples = phi0 + rng.normal(0.0, phi_std, size=(num_samples,)).astype(np.float32)
            phi_samples = np.clip(phi_samples, 0.0, np.float32(math.pi))

        if metric_active:
            r_samples = rng.normal(float(question_dist), sigma_m, size=(num_samples,)).astype(np.float32)
            r_samples = np.clip(r_samples, 0.05, None)
        else:
            r_lo, r_hi = NO_METRIC_RADIAL_PROPOSAL_RANGE_M
            r_samples = rng.uniform(r_lo, r_hi, size=(num_samples,)).astype(np.float32)

        st = np.sin(phi_samples)
        dirs = np.stack(
            [
                np.cos(theta_samples) * st,
                np.cos(phi_samples),
                np.sin(theta_samples) * st,
            ],
            axis=1,
        ).astype(np.float32)
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

        pts = anchor_pos_hab[None, :] + (r_samples[:, None] * dirs)

        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        zs = pts[:, 2].astype(np.float32)

        pdf_params = {k: v for k, v in params.items() if not str(k).startswith("_")}
        logps = _combined_logpdf(xs, ys, zs, pdf_params).astype(np.float64)

        m = float(np.max(logps))
        ws = np.exp(logps - m)
        wsum = float(np.sum(ws) + 1e-12)
        wnorm = ws / wsum

        mean_xyz = (pts.astype(np.float64) * wnorm[:, None]).sum(axis=0).astype(np.float32)

        imax = int(np.argmax(logps))
        map_xyz = pts[imax].astype(np.float32)
        map_logp = float(logps[imax])

        ess = float(1.0 / (np.sum(wnorm * wnorm) + 1e-12))

        dif = pts.astype(np.float64) - mean_xyz.astype(np.float64)[None, :]
        var = (wnorm[:, None] * (dif * dif)).sum(axis=0)
        std_xyz = np.sqrt(var).astype(np.float32)

        chosen_xyz = map_xyz if use_map else mean_xyz

        chosen_logp = float(
            _combined_logpdf(
                np.array([chosen_xyz[0]], dtype=np.float32),
                np.array([chosen_xyz[1]], dtype=np.float32),
                np.array([chosen_xyz[2]], dtype=np.float32),
                pdf_params,
            )[0]
        )

        return {
            "xyz_mean_hab": mean_xyz.tolist(),
            "xyz_map_hab": map_xyz.tolist(),
            "xyz_chosen_hab": chosen_xyz.tolist(),
            "chosen_logp": float(chosen_logp),
            "map_logp": float(map_logp),
            "ess": float(ess),
            "std_xyz": std_xyz.tolist(),
            "num_samples": int(num_samples),
            "planar": bool(planar),
            "theta_std": float(theta_std),
            "sigma_m_used": float(sigma_m),
            "sigma_s_used": float(pdf_params["sigma_s"]),
            "kappa_used": float(kappa),
            "anchor_size_used": (anchor_size or [0.5, 0.5, 0.5]),
            "anchor_max_dim_used": float(params.get("_anchor_max_dim", 0.5)),
            "flatten_semantic": bool(flatten_semantic_for_where),
            "question_dist_m": (float(question_dist) if question_dist is not None else None),
            "metric_kernel_active": metric_active,
            "radial_proposal": (
                "normal(d0, sigma_m)" if metric_active
                else "uniform%s (proposal support only; no radial kernel)"
                % (NO_METRIC_RADIAL_PROPOSAL_RANGE_M,)
            ),
        }
