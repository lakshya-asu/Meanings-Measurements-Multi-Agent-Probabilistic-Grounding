#!/usr/bin/env python3
# /home/artemis/project/graph_eqa_swagat/src/planners/vlm_planner_msp.py
#TODO: Engine needs to be reworked

from __future__ import annotations

import os
import json
import math
import base64
import mimetypes
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import google.generativeai as genai

# Graph EQA / Habitat imports
from src.envs.utils import pos_normal_to_habitat
from src.utils.data_utils import get_latest_image

# MSP imports
from src.msp.pdf import combined_logpdf as _combined_logpdf


# =============================================================================
# Config / Setup
# =============================================================================

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("GOOGLE_API_KEY must be set in the environment.")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Keep a single global model instance
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")


# =============================================================================
# IO helpers
# =============================================================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_latest_image(out_path: Path) -> Optional[str]:
    img = get_latest_image(Path(out_path))
    return str(img) if img else None


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[MSP] Failed to write jsonl log {path}: {e}")


def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[MSP] Failed to write json {path}: {e}")


# =============================================================================
# Trace helpers
# =============================================================================

def _write_trace_step(out_dir: Path, t: int, trace: Dict[str, Any]) -> None:
    _write_json(out_dir / f"trace_step_{t:03d}.json", trace)


def _append_trace_txt(out_dir: Path, lines: List[str]) -> None:
    try:
        with open(out_dir / "llm_outputs_smart.txt", "a") as f:
            for ln in lines:
                f.write(ln.rstrip() + "\n")
            f.write("\n")
    except Exception as e:
        print(f"[TRACE] Could not append trace txt: {e}")


def _shorten(s: str, n: int = 260) -> str:
    s = str(s or "")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _summarize_rank_delta(scored: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    top = scored[:k]
    gap = None
    if len(top) >= 2:
        gap = float(top[0]["msp_score"] - top[1]["msp_score"])
    return {
        "topk": [
            {
                "id": str(x.get("id", "")),
                "name": x.get("name", ""),
                "score": float(x.get("msp_score", 0.0)),
                "pos_hab": x.get("position", None),
            }
            for x in top
        ],
        "gap_1_2": gap,
    }


# =============================================================================
# Math / geometry helpers
# =============================================================================

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [0, 2π)."""
    two_pi = 2.0 * math.pi
    return (angle % two_pi + two_pi) % two_pi


def _camera_theta_to_world(vlm_theta: float, agent_yaw: float) -> float:
    """
    Egocentric camera frame:
      0.0 rad = straight ahead (center of image)
      +π/2    = left of image
      -π/2    = right of image
      π       = behind

    Convert to world by adding agent yaw.
    """
    return _wrap_angle(agent_yaw + float(vlm_theta))


def _parse_q_dist(question: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", (question or "").lower())
    return float(m.group(1)) if m else 1.0


def _unit_dir_from_theta_phi(theta: float, phi: float) -> np.ndarray:
    """
    Spherical -> unit vector (Hab/world):
      x = cos(theta)*sin(phi)
      y = cos(phi)
      z = sin(theta)*sin(phi)
    With phi=pi/2 -> level plane (y=0).
    """
    st = math.sin(phi)
    v = np.array(
        [math.cos(theta) * st, math.cos(phi), math.sin(theta) * st],
        dtype=np.float32,
    )
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n


# =============================================================================
# STEP 1: Spatial kernel (VLM ONLY; no fallback priors)
# =============================================================================

def get_vlm_spatial_kernel_params(
    image_path: Optional[str],
    question: str,
    anchor_name: str,
    anchor_pos_hab: np.ndarray,
    agent_pos_hab: np.ndarray,
    agent_yaw: float,
    anchor_front_yaw_world: Optional[float] = None,  # kept for logging only
    log_jsonl_path: Optional[Path] = None,
    step_t: Optional[int] = None,
) -> Dict[str, Any]:
    """
    VLM-only kernel.

    Returns dict:
      ok: bool
      (if ok) theta, phi, kappa, reasoning, debug
      (if not ok) error, debug
    """
    # normalize (kept to avoid accidental type issues)
    _ = np.asarray(anchor_pos_hab, dtype=np.float32)
    _ = np.asarray(agent_pos_hab, dtype=np.float32)

    # Hard fail if no image
    if not image_path or not os.path.exists(str(image_path)):
        out = {
            "ok": False,
            "error": "No image available for VLM kernel; no fallback allowed.",
            "debug": {
                "used_vlm": False,
                "image_path": image_path,
                "anchor_front_yaw_world": (
                    float(anchor_front_yaw_world) if anchor_front_yaw_world is not None else None
                ),
            },
        }
        if log_jsonl_path:
            _write_jsonl(
                log_jsonl_path,
                {
                    "type": "kernel_fail_no_image",
                    "t": step_t,
                    "image_path": image_path,
                    "question": question,
                    "anchor_name": anchor_name,
                    "result": out,
                },
            )
        return out

    sys_prompt = """
SYSTEM: You are a Geometric Orientation Engine.

YOUR GOAL:
Identify the **INTRINSIC FRONT VECTOR** of the Reference Object relative to the Camera.

CRITICAL RULES:
1. IGNORE DISTANCE (e.g., "3 meters"). Only orientation.
2. DO NOT look for the target point. Only object-facing direction.
3. Output only face orientation (functional front) of the object.

INTRINSIC FRONT examples:
- Sofa/Chair: direction your knees point when seated.
- TV/Monitor: screen-facing direction.
- Fridge/Cabinet: door/drawer opening face.

CAMERA COORDINATES (Egocentric, top-down):
THETA (azimuth):
  0.00 rad  = Straight ahead (center of image)
  +1.57 rad = LEFT of image
  -1.57 rad (or 4.71) = RIGHT of image
  3.14 rad  = behind camera

PHI (elevation):
  1.57 rad = level
  0.00 rad = above / on top
  3.14 rad = below / under

Return JSON only.
"""

    schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING),
            "theta_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "phi_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "kappa": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        },
        required=["reasoning", "theta_radians", "phi_radians", "kappa"],
    )

    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"

    # IMPORTANT: we intentionally do NOT show the full question to avoid distance bias.
    sanitized_query = f"Where is the intrinsic front of the {anchor_name}?"

    prior_hint = ""
    if anchor_front_yaw_world is not None:
        prior_hint = (
            f"\nNOTE (for reference only): dataset provides anchor-front yaw (WORLD rad): "
            f"{float(anchor_front_yaw_world):.4f}\n"
        )

    messages = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        f"{sys_prompt}\n"
                        f"{prior_hint}\n"
                        f"Reference Object: {anchor_name}\n"
                        f"Task: {sanitized_query}\n"
                        "Instruction: Output Theta/Phi direction of that face relative to the camera."
                    )
                },
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": encode_image(str(image_path)),
                    }
                },
            ],
        }
    ]

    raw_text = ""
    try:
        resp = gemini_model.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                response_schema=schema,
            ),
        )
        raw_text = resp.text
        d = json.loads(resp.text)

        theta_cam = float(d["theta_radians"])
        phi = float(d["phi_radians"])
        kappa = float(d["kappa"])
        reasoning = d.get("reasoning", "")

        if not math.isfinite(kappa) or kappa <= 0.1:
            kappa = 5.0

        theta_world = _camera_theta_to_world(theta_cam, agent_yaw)

        print(
            f"[MSP-VLM-ANGLE] question='{question}', anchor='{anchor_name}', "
            f"agent_yaw={agent_yaw:.3f}, theta_cam={theta_cam:.3f}, "
            f"theta_world={theta_world:.3f}, phi={phi:.3f}, kappa={kappa:.3f}"
        )

        out = {
            "ok": True,
            "theta": float(theta_world),
            "phi": float(phi),
            "kappa": float(kappa),
            "reasoning": reasoning,
            "debug": {
                "theta_cam": float(theta_cam),
                "theta_world": float(theta_world),
                "agent_yaw": float(agent_yaw),
                "used_vlm": True,
                "anchor_front_yaw_world": (
                    float(anchor_front_yaw_world) if anchor_front_yaw_world is not None else None
                ),
            },
        }

        if log_jsonl_path:
            _write_jsonl(
                log_jsonl_path,
                {
                    "type": "kernel",
                    "t": step_t,
                    "image_path": image_path,
                    "question": question,
                    "anchor_name": anchor_name,
                    "used_image": True,
                    "messages": messages[0]["parts"][0]["text"],
                    "raw_response_text": raw_text,
                    "parsed": d,
                    "result": out,
                },
            )

        return out

    except Exception as e:
        out = {
            "ok": False,
            "error": f"VLM kernel call failed; no fallback allowed. Error: {e}",
            "debug": {
                "used_vlm": False,
                "raw_response_text": raw_text,
                "image_path": image_path,
                "anchor_front_yaw_world": (
                    float(anchor_front_yaw_world) if anchor_front_yaw_world is not None else None
                ),
            },
        }
        if log_jsonl_path:
            _write_jsonl(
                log_jsonl_path,
                {
                    "type": "kernel_fail_vlm_error",
                    "t": step_t,
                    "image_path": image_path,
                    "question": question,
                    "anchor_name": anchor_name,
                    "raw_response_text": raw_text,
                    "error": str(e),
                    "result": out,
                },
            )
        return out


# =============================================================================
# STEP 2: MSP scoring core
# =============================================================================

class MSPEngineSmart:
    def __init__(self) -> None:
        pass

    def _get_metric_semantic_params(
        self,
        anchor_pos_hab: np.ndarray,
        candidate_pos_hab: np.ndarray,
        candidate_size: Optional[List[float]],
        distance_m: float,
    ) -> Dict[str, float]:
        pos = np.asarray(anchor_pos_hab, dtype=np.float32)
        size = candidate_size or [0.5, 0.5, 0.5]
        w, d, h = [float(x) for x in size[:3]]
        max_dim = max(w, d, h)

        return {
            "mu_x": float(pos[0]),
            "mu_y": float(pos[1]),
            "mu_z": float(pos[2]),
            "sigma_s": 0.5 * max_dim,
            "x0": float(anchor_pos_hab[0]),
            "y0": float(anchor_pos_hab[1]),
            "z0": float(anchor_pos_hab[2]),
            "d0": float(distance_m),
            "sigma_m": 0.3 * max_dim,
        }

    def score_point(
        self,
        point_hab: np.ndarray,
        anchor_pos_hab: np.ndarray,
        kernel_params: Dict[str, Any],
        question_dist: float,
        candidate_size: Optional[List[float]] = None,
    ) -> float:
        point_hab = np.asarray(point_hab, dtype=np.float32)
        anchor_pos_hab = np.asarray(anchor_pos_hab, dtype=np.float32)

        params = {
            **self._get_metric_semantic_params(
                anchor_pos_hab=anchor_pos_hab,
                candidate_pos_hab=point_hab,
                candidate_size=candidate_size or [0.5, 0.5, 0.5],
                distance_m=question_dist,
            ),
            "theta0": float(kernel_params["theta"]),
            "phi0": float(kernel_params["phi"]),
            "kappa": float(kernel_params["kappa"]),
        }

        logp = float(
            _combined_logpdf(
                np.array([point_hab[0]]),
                np.array([point_hab[1]]),
                np.array([point_hab[2]]),
                params,
            )[0]
        )
        return logp

    def score_candidates(
        self,
        objects: List[Dict[str, Any]],
        frontiers: List[Dict[str, Any]],
        anchor_pos_hab: np.ndarray,
        kernel_params: Dict[str, Any],
        question_dist: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        anchor_pos_hab = np.asarray(anchor_pos_hab, dtype=np.float32)

        scored_objects: List[Dict[str, Any]] = []
        for obj in objects:
            pos = np.asarray(obj["position"], dtype=np.float32)
            params = {
                **self._get_metric_semantic_params(anchor_pos_hab, pos, obj.get("size", None), question_dist),
                "theta0": float(kernel_params["theta"]),
                "phi0": float(kernel_params["phi"]),
                "kappa": float(kernel_params["kappa"]),
            }
            logp = float(
                _combined_logpdf(
                    np.array([pos[0]]),
                    np.array([pos[1]]),
                    np.array([pos[2]]),
                    params,
                )[0]
            )
            scored_objects.append({**obj, "msp_score": logp})

        scored_frontiers: List[Dict[str, Any]] = []
        for fr in frontiers:
            pos = np.asarray(fr["position"], dtype=np.float32)
            params = {
                **self._get_metric_semantic_params(anchor_pos_hab, pos, fr.get("size", None), question_dist),
                "theta0": float(kernel_params["theta"]),
                "phi0": float(kernel_params["phi"]),
                "kappa": float(kernel_params["kappa"]),
            }
            logp = float(
                _combined_logpdf(
                    np.array([pos[0]]),
                    np.array([pos[1]]),
                    np.array([pos[2]]),
                    params,
                )[0]
            )
            scored_frontiers.append({**fr, "msp_score": logp})

        scored_objects.sort(key=lambda x: x["msp_score"], reverse=True)
        scored_frontiers.sort(key=lambda x: x["msp_score"], reverse=True)
        return scored_objects, scored_frontiers


# =============================================================================
# STEP 3: Planner — VLM sees scored candidates + point guess, logs everything
# =============================================================================

class VLMPlannerMSP_Smart:
    def __init__(self, cfg, sg_sim, question, gt=None, out_path=".", **kwargs):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self._question = question
        self._out_path = Path(out_path)
        self._t = 0
        self._history = ""
        self._outputs_to_save = [f"Question: {question}\n"]

        raw_answer_mode = str(getattr(cfg, "answer_mode", "") or "").lower().strip()

        nested_mode = ""
        try:
            nested_mode = str(getattr(getattr(cfg, "msp_nobnn", None), "mode", "") or "").lower().strip()
        except Exception:
            nested_mode = ""

        def _normalize_mode(m: str) -> str:
            m = (m or "").lower().strip()
            mapping = {
                "msp_point": "where",
                "msp_where": "where",
                "point": "where",
                "where": "where",
                "msp_object": "which",
                "msp_which": "which",
                "object": "which",
                "which": "which",
                "eqa": "which",
            }
            return mapping.get(m, m)

        resolved = _normalize_mode(nested_mode) if nested_mode else _normalize_mode(raw_answer_mode)
        if resolved not in ["where", "which"]:
            resolved = "where"
        self.answer_mode: str = resolved

        self._anchor_label: Optional[str] = kwargs.get("anchor_label", None)
        self._anchor_center_hab: Optional[np.ndarray] = kwargs.get("anchor_center_hab", None)
        if self._anchor_center_hab is not None:
            self._anchor_center_hab = np.asarray(self._anchor_center_hab, dtype=np.float32)

        # kept for logging / cues only (kernel will NOT fallback to this)
        self._anchor_front_yaw_world: Optional[float] = kwargs.get("anchor_front_yaw_world", None)
        if self._anchor_front_yaw_world is not None:
            self._anchor_front_yaw_world = float(self._anchor_front_yaw_world)

        self.msp_engine = MSPEngineSmart()

        self._vlm_calls_path = self._out_path / "vlm_calls.jsonl"

        print(f"\n[MSP SMART INIT] mode={self.answer_mode} Q: {self._question}")
        print(f"  - Anchor label hint: {self._anchor_label}")
        print(f"  - Anchor center hint (hab): {self._anchor_center_hab}")
        print(f"  - Anchor front yaw world (intrinsic): {self._anchor_front_yaw_world}")

        _write_jsonl(
            self._vlm_calls_path,
            {
                "type": "planner_init",
                "mode": self.answer_mode,
                "question": self._question,
                "anchor_label_hint": self._anchor_label,
                "anchor_center_hint": (
                    self._anchor_center_hab.tolist() if self._anchor_center_hab is not None else None
                ),
                "anchor_front_yaw_world": self._anchor_front_yaw_world,
            },
        )

        try:
            with open(self._out_path / "llm_outputs_smart.txt", "w") as f:
                f.write(f"Question: {self._question}\n\n")
        except Exception:
            pass

    @property
    def t(self) -> int:
        return self._t

    def _get_scene_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        objects: List[Dict[str, Any]] = []
        for oid, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            try:
                pos_norm = self.sg_sim.get_position_from_id(oid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)), dtype=np.float32)
                objects.append(
                    {
                        "id": str(oid),
                        "name": str(name).lower(),
                        "position": pos_hab.tolist(),
                        "type": "object",
                        "size": [0.5, 0.5, 0.5],
                    }
                )
            except Exception:
                continue

        frontiers: List[Dict[str, Any]] = []
        for fid in getattr(self.sg_sim, "frontier_node_ids", []) or []:
            try:
                pos_norm = self.sg_sim.get_position_from_id(fid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)), dtype=np.float32)
                frontiers.append(
                    {
                        "id": str(fid),
                        "name": "frontier",
                        "position": pos_hab.tolist(),
                        "type": "frontier",
                        "size": [0.5, 0.5, 0.5],
                    }
                )
            except Exception:
                continue

        return objects, frontiers

    # -------------------------------------------------------------------------
    # Anchor resolution + debug trace
    # -------------------------------------------------------------------------

    def _resolve_anchor(
        self, objects: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        dbg: Dict[str, Any] = {
            "anchor_label_hint": self._anchor_label,
            "anchor_center_hab_hint": (
                self._anchor_center_hab.tolist() if self._anchor_center_hab is not None else None
            ),
            "match_strategy": None,
            "num_objects": len(objects),
            "candidates": [],
            "chosen": None,
        }

        if not self._anchor_label:
            dbg["match_strategy"] = "no_label_hint"
            return None, "unknown_anchor", dbg

        label = self._anchor_label.strip().lower()

        candidates = [o for o in objects if label in (o.get("name") or "")]
        dbg["match_strategy"] = "substring_label"
        if not candidates:
            label2 = label.replace(" ", "")
            candidates = [o for o in objects if label2 in (o.get("name") or "").replace(" ", "")]
            dbg["match_strategy"] = "substring_label_no_spaces"

        if not candidates:
            dbg["match_strategy"] = str(dbg["match_strategy"]) + "_no_match"
            return None, label, dbg

        for o in candidates[:30]:
            dbg["candidates"].append(
                {"id": str(o.get("id", "")), "name": o.get("name", ""), "pos_hab": o.get("position", None)}
            )

        if self._anchor_center_hab is None:
            best = candidates[0]
            dbg["chosen"] = {"id": str(best.get("id", "")), "name": best.get("name", ""), "pos_hab": best.get("position")}
            return best, best.get("name", label), dbg

        c0 = np.asarray(self._anchor_center_hab, dtype=np.float32)
        best = None
        best_d = 1e9
        for o in candidates:
            p = np.asarray(o["position"], dtype=np.float32)
            d = float(np.linalg.norm(p - c0))
            if d < best_d:
                best_d = d
                best = o

        dbg["match_strategy"] = str(dbg["match_strategy"]) + "_closest_to_csv_center"
        dbg["chosen"] = {
            "id": str(best.get("id", "")) if best else "",
            "name": best.get("name", "") if best else "",
            "pos_hab": best.get("position") if best else None,
            "dist_to_center": float(best_d) if best else None,
        }
        return best, best.get("name", label) if best else label, dbg

    # -------------------------------------------------------------------------
    # Selector prompt / call
    # -------------------------------------------------------------------------

    def _build_selector_prompt(
        self,
        agent_state: str,
        anchor_name: str,
        anchor_pos_hab: np.ndarray,
        kernel: Dict[str, Any],
        dist_m: float,
        top_objects: List[Dict[str, Any]],
        top_frontiers: List[Dict[str, Any]],
        point_guess: Optional[Dict[str, Any]],
    ) -> str:
        if self.answer_mode == "where":
            mode_rules = (
                "MODE=WHERE: You may choose a coordinate point (POINT_GUESS) OR an object id.\n"
                "If the query is about a location in space, POINT_GUESS is usually correct.\n"
            )
        else:
            mode_rules = (
                "MODE=WHICH: You MUST choose an OBJECT from TOP OBJECTS.\n"
                "POINT_GUESS is NOT allowed.\n"
                "If nothing matches, choose the best available object with low confidence and explain.\n"
            )

        obj_lines = []
        for o in top_objects:
            p = o.get("position", [0, 0, 0])
            obj_lines.append(
                f"- id={o['id']} name={o.get('name','')} score={o.get('msp_score',0.0):.3f} "
                f"center_xyz_hab=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            )

        fr_lines = []
        for f in top_frontiers:
            p = f.get("position", [0, 0, 0])
            fr_lines.append(
                f"- id={f['id']} score={f.get('msp_score',0.0):.3f} xyz_hab=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            )

        point_block = "POINT_GUESS: none\n"
        if point_guess is not None:
            pg = point_guess["target_xyz_hab"]
            point_block = (
                f"POINT_GUESS: id=POINT_GUESS score={point_guess['msp_score']:.3f} "
                f"xyz_hab=[{pg[0]:.3f},{pg[1]:.3f},{pg[2]:.3f}]\n"
            )

        return f"""
You are a selector for a robot spatial query system.

{mode_rules}

Question: {self._question}

Anchor:
- name: {anchor_name}
- anchor_xyz_hab: [{anchor_pos_hab[0]:.3f},{anchor_pos_hab[1]:.3f},{anchor_pos_hab[2]:.3f}]
- requested_distance_m: {dist_m:.3f}

Spatial Kernel (world):
- theta_world: {kernel['theta']:.4f}
- phi: {kernel['phi']:.4f}
- kappa: {kernel['kappa']:.3f}
- reasoning: {kernel.get('reasoning','')}

Candidates (ranked by MSP logpdf):
{point_block}

TOP OBJECTS:
{chr(10).join(obj_lines) if obj_lines else "- none"}

TOP FRONTIERS:
{chr(10).join(fr_lines) if fr_lines else "- none"}

History:
{self._history}

Current state:
{agent_state}

Task:
Decide whether to:
- answer now (and what the answer is),
- or move (goto_object/goto_frontier),
- or lookaround.

Output STRICT JSON only:
- thought: string
- action_type: one of ["goto_frontier","goto_object","lookaround","answer"]
- chosen_id: string
    * WHERE: "POINT_GUESS" or an object id (from TOP OBJECTS)
    * WHICH: must be an object id from TOP OBJECTS
- target_xyz_hab: [x,y,z] only if chosen_id=="POINT_GUESS", else []
- answer_text: string
- confidence: float in [0,1]
"""

    def _call_selector_llm(self, prompt: str) -> Tuple[Dict[str, Any], str]:
        selector_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "thought": genai.protos.Schema(type=genai.protos.Type.STRING),
                "action_type": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=["goto_frontier", "goto_object", "lookaround", "answer"],
                ),
                "chosen_id": genai.protos.Schema(type=genai.protos.Type.STRING),
                "target_xyz_hab": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.NUMBER),
                ),
                "answer_text": genai.protos.Schema(type=genai.protos.Type.STRING),
                "confidence": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            },
            required=["thought", "action_type", "chosen_id", "confidence"],
        )

        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        resp = gemini_model.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
                response_schema=selector_schema,
            ),
        )
        return json.loads(resp.text), resp.text

    # -------------------------------------------------------------------------
    # Room constraint hook (placeholder)
    # -------------------------------------------------------------------------

    def _get_anchor_room_id(self, anchor_object_id: str) -> Optional[str]:
        return None

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def get_next_action(self, agent_yaw_rad: float = 0.0, agent_pos_hab: Optional[np.ndarray] = None):
        if agent_pos_hab is None:
            agent_pos_hab = np.array([0, 0, 0], dtype=np.float32)
        else:
            agent_pos_hab = np.asarray(agent_pos_hab, dtype=np.float32)

        objects, frontiers = self._get_scene_data()
        img_path = _safe_latest_image(self._out_path)

        anchor_obj, anchor_name, anchor_dbg = self._resolve_anchor(objects)

        # Anchor missing => explore
        if anchor_obj is None:
            _write_jsonl(
                self._vlm_calls_path,
                {
                    "type": "no_anchor",
                    "t": self._t,
                    "question": self._question,
                    "mode": self.answer_mode,
                    "anchor_resolution": anchor_dbg,
                    "num_objects": len(objects),
                    "num_frontiers": len(frontiers),
                },
            )

            trace = {
                "t": int(self._t),
                "mode": self.answer_mode,
                "question": self._question,
                "agent": {"pos_hab": agent_pos_hab.tolist(), "yaw_world_rad": float(agent_yaw_rad)},
                "anchor_resolution": anchor_dbg,
                "anchor": None,
                "kernel": None,
                "scoring": None,
                "selector_output": None,
                "guardrails": None,
                "decision_rationale": {"recommended": "explore", "note": "Anchor not resolved."},
                "room_constraint": {"anchor_room_id": None, "room_constraint_enforced": False},
            }
            _write_trace_step(self._out_path, self._t, trace)

            txt = [
                f"--- STEP {self._t} ---",
                f"Q: {self._question}",
                f"Mode: {self.answer_mode}",
                f"Agent: pos_hab={agent_pos_hab.tolist()} yaw={float(agent_yaw_rad):.3f} rad",
                "Anchor resolution FAILED:",
                f"- label_hint={anchor_dbg.get('anchor_label_hint')}",
                f"- strategy={anchor_dbg.get('match_strategy')}",
                f"- candidates={len(anchor_dbg.get('candidates', []))}",
            ]

            if len(frontiers) == 0:
                plan = {
                    "thought": "No anchor resolved and no frontiers available. Lookaround.",
                    "action_type": "lookaround",
                    "chosen_id": "",
                    "target_xyz_hab": [],
                    "answer_text": "",
                    "confidence": 0.0,
                }
                txt.append("Decision: lookaround (no frontiers).")
                _append_trace_txt(self._out_path, txt)

                self._history += f"[t={self._t}] action=lookaround\n"
                self._t += 1
                return None, None, False, 0.0, plan

            fid = str(frontiers[0]["id"])
            plan = {
                "thought": "Anchor not found yet; explore a frontier.",
                "action_type": "goto_frontier",
                "chosen_id": fid,
                "target_xyz_hab": [],
                "answer_text": "",
                "confidence": 0.0,
            }
            txt.append(f"Decision: goto_frontier chosen={fid} (anchor unresolved).")
            _append_trace_txt(self._out_path, txt)

            self._history += f"[t={self._t}] action=goto_frontier chosen_id={fid}\n"
            self._t += 1
            return self.sg_sim.get_position_from_id(fid), fid, False, 0.0, plan

        anchor_pos = np.asarray(anchor_obj["position"], dtype=np.float32)

        anchor_room_id = self._get_anchor_room_id(str(anchor_obj.get("id", "")))
        room_trace = {
            "anchor_room_id": anchor_room_id,
            "room_constraint_enforced": False,
            "note": "Room constraint not enforced in this build (no room mapping available).",
        }

        # ---------------------------------------------------------------------
        # Kernel (VLM-only). Hard fail if it fails.
        # ---------------------------------------------------------------------
        kernel = get_vlm_spatial_kernel_params(
            image_path=img_path,
            question=self._question,
            anchor_name=anchor_name,
            anchor_pos_hab=anchor_pos,
            agent_pos_hab=agent_pos_hab,
            agent_yaw=agent_yaw_rad,
            anchor_front_yaw_world=self._anchor_front_yaw_world,
            log_jsonl_path=self._vlm_calls_path,
            step_t=self._t,
        )

        if not kernel.get("ok", False):
            _write_jsonl(
                self._vlm_calls_path,
                {
                    "type": "planner_kernel_failed",
                    "t": self._t,
                    "mode": self.answer_mode,
                    "question": self._question,
                    "kernel": kernel,
                },
            )

            trace = {
                "t": int(self._t),
                "mode": self.answer_mode,
                "question": self._question,
                "agent": {"pos_hab": agent_pos_hab.tolist(), "yaw_world_rad": float(agent_yaw_rad)},
                "anchor_resolution": anchor_dbg,
                "anchor": {"id": str(anchor_obj.get("id", "")), "name": anchor_name, "pos_hab": anchor_pos.tolist()},
                "kernel": kernel,
                "scoring": None,
                "selector_output": None,
                "guardrails": None,
                "decision_rationale": {"recommended": "fail", "note": kernel.get("error", "")},
                "room_constraint": room_trace,
            }
            _write_trace_step(self._out_path, self._t, trace)

            _append_trace_txt(
                self._out_path,
                [
                    f"--- STEP {self._t} ---",
                    f"Q: {self._question}",
                    f"Mode: {self.answer_mode}",
                    f"KERNEL FAILED: {kernel.get('error','unknown error')}",
                    "Decision: fail step (no fallback).",
                ],
            )

            plan = {
                "thought": f"Kernel failed; no fallback allowed. error={kernel.get('error','')}",
                "action_type": "answer",
                "chosen_id": "",
                "target_xyz_hab": [],
                "answer_text": "",
                "confidence": 0.0,
            }

            self._history += f"[t={self._t}] kernel_failed error={kernel.get('error','')}\n"
            self._t += 1
            return None, None, False, 0.0, plan

        dist_m = _parse_q_dist(self._question)

        # ---------------------------------------------------------------------
        # MSP scoring
        # ---------------------------------------------------------------------
        msp_objects, msp_frontiers = self.msp_engine.score_candidates(
            objects=objects,
            frontiers=frontiers,
            anchor_pos_hab=anchor_pos,
            kernel_params=kernel,
            question_dist=dist_m,
        )

        # WHERE point guess always computed/logged (selector can ignore)
        dir_world = _unit_dir_from_theta_phi(float(kernel["theta"]), float(kernel["phi"]))
        point_xyz = (anchor_pos + float(dist_m) * dir_world).astype(np.float32)

        point_logp = self.msp_engine.score_point(
            point_hab=point_xyz,
            anchor_pos_hab=anchor_pos,
            kernel_params=kernel,
            question_dist=dist_m,
            candidate_size=[0.5, 0.5, 0.5],
        )

        point_guess = {"id": "POINT_GUESS", "target_xyz_hab": point_xyz.tolist(), "msp_score": float(point_logp)}

        K_OBJ = int(getattr(self.cfg, "selector_topk_objects", 12))
        K_FR = int(getattr(self.cfg, "selector_topk_frontiers", 8))
        top_objects = msp_objects[:K_OBJ]
        top_frontiers = msp_frontiers[:K_FR]

        selector_context = {
            "t": self._t,
            "mode": self.answer_mode,
            "question": self._question,
            "agent": {"pos_hab": agent_pos_hab.tolist(), "yaw_world_rad": float(agent_yaw_rad)},
            "anchor_resolution": anchor_dbg,
            "anchor": {
                "id": anchor_obj["id"],
                "name": anchor_name,
                "pos_hab": anchor_pos.tolist(),
                "anchor_front_yaw_world": self._anchor_front_yaw_world,
                "anchor_room_id": anchor_room_id,
            },
            "kernel": kernel,
            "dist_m": dist_m,
            "point_guess": point_guess,
            "top_objects": [
                {"id": o["id"], "name": o.get("name", ""), "msp_score": float(o.get("msp_score", 0.0)), "pos_hab": o.get("position")}
                for o in top_objects
            ],
            "top_frontiers": [
                {"id": f["id"], "msp_score": float(f.get("msp_score", 0.0)), "pos_hab": f.get("position")}
                for f in top_frontiers
            ],
        }
        selector_context_path = self._out_path / f"selector_context_step_{self._t}.json"
        _write_json(selector_context_path, selector_context)

        # ---------------------------------------------------------------------
        # Selector VLM
        # ---------------------------------------------------------------------
        agent_state_str = self.sg_sim.get_current_semantic_state_str()
        prompt = self._build_selector_prompt(
            agent_state=agent_state_str,
            anchor_name=anchor_name,
            anchor_pos_hab=anchor_pos,
            kernel=kernel,
            dist_m=dist_m,
            top_objects=top_objects,
            top_frontiers=top_frontiers,
            point_guess=point_guess,
        )

        raw_text = ""
        try:
            plan, raw_text = self._call_selector_llm(prompt)
            _write_jsonl(
                self._vlm_calls_path,
                {"type": "selector", "t": self._t, "mode": self.answer_mode, "prompt": prompt, "raw_response_text": raw_text, "parsed": plan},
            )
        except Exception as e:
            _write_jsonl(
                self._vlm_calls_path,
                {"type": "selector_error", "t": self._t, "mode": self.answer_mode, "error": str(e), "raw_response_text": raw_text},
            )

            # Selector fallback (kept) — kernel fallback removed, selector fallback still OK.
            if len(top_objects) > 0:
                plan = {
                    "thought": f"Selector failed; fallback to best object. Error={e}",
                    "action_type": "answer" if self.answer_mode == "which" else "goto_object",
                    "chosen_id": str(top_objects[0]["id"]),
                    "target_xyz_hab": [],
                    "answer_text": f"Fallback best object: {top_objects[0].get('name','')}",
                    "confidence": 0.2,
                }
            elif len(top_frontiers) > 0:
                plan = {
                    "thought": f"Selector failed; fallback to best frontier. Error={e}",
                    "action_type": "goto_frontier",
                    "chosen_id": str(top_frontiers[0]["id"]),
                    "target_xyz_hab": [],
                    "answer_text": "",
                    "confidence": 0.0,
                }
            else:
                plan = {
                    "thought": f"Selector failed; lookaround. Error={e}",
                    "action_type": "lookaround",
                    "chosen_id": "",
                    "target_xyz_hab": [],
                    "answer_text": "",
                    "confidence": 0.0,
                }

        # ---------------------------------------------------------------------
        # Guardrails
        # ---------------------------------------------------------------------
        chosen_id = str(plan.get("chosen_id", "")).strip()
        action = str(plan.get("action_type", "goto_frontier")).strip()

        allowed_object_ids = {str(o["id"]) for o in top_objects}
        allowed_frontier_ids = {str(f["id"]) for f in top_frontiers}

        if self.answer_mode == "which":
            if chosen_id == "POINT_GUESS" or chosen_id not in allowed_object_ids:
                if len(top_objects) > 0:
                    forced = top_objects[0]
                    plan["thought"] = f"[guardrail WHICH] Forced to best object because chosen_id={chosen_id} invalid. " + plan.get("thought", "")
                    plan["chosen_id"] = str(forced["id"])
                    plan["target_xyz_hab"] = []
                    chosen_id = str(forced["id"])
                    if action == "answer":
                        plan["answer_text"] = plan.get("answer_text", "") or f"{forced.get('name','object')} (id={forced['id']})"
                    else:
                        plan["action_type"] = "goto_object"
                        action = "goto_object"
                else:
                    plan["thought"] = "[guardrail WHICH] No objects available; switching to lookaround."
                    plan["action_type"] = "lookaround"
                    plan["chosen_id"] = ""
                    plan["target_xyz_hab"] = []
                    action = "lookaround"
                    chosen_id = ""
        else:
            if chosen_id == "POINT_GUESS":
                plan["target_xyz_hab"] = point_guess["target_xyz_hab"]
            else:
                if chosen_id and (chosen_id not in allowed_object_ids) and (chosen_id not in allowed_frontier_ids):
                    if len(top_frontiers) > 0:
                        plan["thought"] = f"[guardrail WHERE] Invalid chosen_id={chosen_id}. Forced to best frontier."
                        plan["action_type"] = "goto_frontier"
                        plan["chosen_id"] = str(top_frontiers[0]["id"])
                        plan["target_xyz_hab"] = []
                        action = "goto_frontier"
                        chosen_id = str(top_frontiers[0]["id"])

        # ---------------------------------------------------------------------
        # Convert chosen to target_pose
        # ---------------------------------------------------------------------
        target_pose = None
        target_id = None

        if action in ("goto_object", "goto_frontier"):
            target_id = chosen_id
            if target_id:
                try:
                    target_pose = self.sg_sim.get_position_from_id(target_id)
                except Exception as e:
                    print(f"[MSP] Failed to get pose for id {target_id}: {e}")

        # ---------------------------------------------------------------------
        # Confidence / traces
        # ---------------------------------------------------------------------
        conf = float(plan.get("confidence", 0.0))
        is_answer = (action == "answer")

        best_obj = top_objects[0] if len(top_objects) > 0 else None

        # Keep selector introspection (useful for debugging)
        plan["selector"] = {
            "mode": self.answer_mode,
            "chosen_id": plan.get("chosen_id", ""),
            "answer_type": ("point" if plan.get("chosen_id", "") == "POINT_GUESS" else "object"),
            "confidence": conf,
            "point_guess": point_guess,
            "best_object": (
                {
                    "id": str(best_obj["id"]),
                    "name": best_obj.get("name", ""),
                    "msp_score": float(best_obj.get("msp_score", 0.0)),
                    "target_xyz_hab": best_obj.get("position"),
                } if best_obj else None
            ),
            "topk_objects": [
                {"id": str(o["id"]), "name": o.get("name", ""), "msp_score": float(o.get("msp_score", 0.0)), "target_xyz_hab": o.get("position")}
                for o in top_objects[:8]
            ],
            "topk_frontiers": [
                {"id": str(f["id"]), "msp_score": float(f.get("msp_score", 0.0)), "target_xyz_hab": f.get("position")}
                for f in top_frontiers[:6]
            ],
        }

        kernel_trace = {
            "ok": True,
            "kernel_theta_world": float(kernel.get("theta", 0.0)),
            "kernel_phi": float(kernel.get("phi", math.pi / 2.0)),
            "kappa": float(kernel.get("kappa", 0.0)),
            "reasoning": kernel.get("reasoning", ""),
            "debug": kernel.get("debug", {}),
            "image_path": img_path,
        }

        score_trace = {
            "dist_m": float(dist_m),
            "point_guess": {"xyz_hab": point_xyz.tolist(), "score": float(point_logp)},
            "objects": _summarize_rank_delta(msp_objects, k=8),
            "frontiers": _summarize_rank_delta(msp_frontiers, k=6),
        }

        obj_gap = score_trace["objects"]["gap_1_2"]
        decision_rationale = {
            "object_gap_1_2": obj_gap,
            "recommended": "answer" if (obj_gap is not None and obj_gap > 2.0) else "explore",
            "note": "Heuristic recommendation; selector LLM makes final decision.",
        }

        guardrail_applied = ("[guardrail" in str(plan.get("thought", "")))

        trace = {
            "t": int(self._t),
            "mode": self.answer_mode,
            "question": self._question,
            "agent": {"pos_hab": agent_pos_hab.tolist(), "yaw_world_rad": float(agent_yaw_rad)},
            "anchor_resolution": anchor_dbg,
            "anchor": {"id": str(anchor_obj.get("id", "")), "name": anchor_name, "pos_hab": anchor_pos.tolist()},
            "kernel": kernel_trace,
            "scoring": score_trace,
            "selector_prompt_path": str(selector_context_path),
            "selector_output": {"action_type": action, "chosen_id": chosen_id, "confidence": conf, "answer_text": plan.get("answer_text", "")},
            "guardrails": {"applied": bool(guardrail_applied), "thought": plan.get("thought", "")},
            "decision_rationale": decision_rationale,
            "room_constraint": room_trace,
        }
        _write_trace_step(self._out_path, self._t, trace)

        # Readable narration
        kdbg = kernel_trace.get("debug", {}) or {}

        txt: List[str] = [
            f"--- STEP {self._t} ---",
            f"Q: {self._question}",
            f"Mode: {self.answer_mode}",
            f"Agent: pos_hab={agent_pos_hab.tolist()} yaw={float(agent_yaw_rad):.3f} rad",
            "Anchor resolution:",
            f"- label_hint={anchor_dbg.get('anchor_label_hint')}",
            f"- strategy={anchor_dbg.get('match_strategy')}",
        ]

        cand_list = anchor_dbg.get("candidates", []) or []
        if cand_list:
            txt.append(f"- candidates_found={len(cand_list)} (showing up to 5):")
            for c in cand_list[:5]:
                txt.append(f"  * {c.get('id')} name={c.get('name')} pos={c.get('pos_hab')}")
        txt.append(f"- chosen={anchor_dbg.get('chosen')}")

        txt.extend(
            [
                "Kernel:",
                f"- theta_final(world)={kernel_trace['kernel_theta_world']:.4f} phi={kernel_trace['kernel_phi']:.4f} kappa={kernel_trace['kappa']:.2f}",
                f"- theta_cam={kdbg.get('theta_cam')} agent_yaw={kdbg.get('agent_yaw')}",
                f"- vlm_reasoning: {_shorten(kernel_trace.get('reasoning',''), 260)}",
                "MSP scoring:",
                f"- point_guess xyz={score_trace['point_guess']['xyz_hab']} score={score_trace['point_guess']['score']:.3f}",
                f"- top object gap(1-2)={obj_gap}",
            ]
        )

        for r in score_trace["objects"]["topk"][:3]:
            txt.append(f"  * obj {r['id']} name={r['name']} score={r['score']:.3f} pos={r['pos_hab']}")
        for r in score_trace["frontiers"]["topk"][:2]:
            txt.append(f"  * frontier {r['id']} score={r['score']:.3f} pos={r['pos_hab']}")

        txt.extend(
            [
                "Room constraint:",
                f"- anchor_room_id={room_trace.get('anchor_room_id')} enforced={room_trace.get('room_constraint_enforced')}",
                f"- note={room_trace.get('note')}",
                "Decision:",
                f"- heuristic_recommendation={decision_rationale.get('recommended')} (gap={obj_gap})",
                f"- selector_action={action} chosen_id={chosen_id} conf={conf:.2f}",
            ]
        )

        if plan.get("answer_text"):
            txt.append(f"- answer_text: {plan.get('answer_text')}")
        if guardrail_applied:
            txt.append(f"- guardrail_applied: {_shorten(plan.get('thought',''), 220)}")
        else:
            txt.append(f"- selector_thought: {_shorten(plan.get('thought',''), 220)}")

        _append_trace_txt(self._out_path, txt)

        self._history += (
            f"[t={self._t}] mode={self.answer_mode} action={action} chosen={plan.get('chosen_id','')} "
            f"conf={conf:.2f} kernel(theta={kernel['theta']:.2f},kappa={kernel['kappa']:.1f}) "
            f"pg_score={point_guess['msp_score']:.2f} "
            f"best_obj={(best_obj.get('name','') if best_obj else 'none')} "
            f"best_obj_score={(best_obj.get('msp_score',0.0) if best_obj else 0.0):.2f}\n"
        )

        self._outputs_to_save.append(self._history)
        try:
            with open(self._out_path / "llm_outputs_smart_compact.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception as e:
            print(f"[MSP PLANNER] Could not write compact logs: {e}")

        self._t += 1

        is_confident = is_answer and (conf >= 0.90)
        return target_pose, target_id, is_confident, conf, plan