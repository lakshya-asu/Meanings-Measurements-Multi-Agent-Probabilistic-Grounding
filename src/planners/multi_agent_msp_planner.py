import numpy as np
import json
import click
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.data_utils import get_latest_image
from src.schema.prediction import normalize_prediction

# Import the MSP Engine (numeric-conditioning aware subclass of MSPEngineSmart)
from src.planners.msp_engine_numeric import NumericConditioningEngine, gt_front_theta

# MAPG-01: kernel bandwidths from real Hydra AABB extents (pure helpers,
# host-importable; formulas documented in the module docstring)
from src.msp.kernel_bandwidths import (
    FIXED_SIZE_M,
    resolve_mode as resolve_bandwidth_mode,
    resolve_object_size_hab,
    bandwidth_size_hab,
)

# P0 fix 1: single deterministic source for d0 (no silent 1.0 m default)
from src.parsing.metric_literal import (
    parse_metric_literal,
    infer_relation,
    resolve_categorical_distance,
)

# P0 fix 3: programmatic verifier checks
from src.verification.checks import run_checks, failed_reasons

# MAPG-12: navmesh masking + renormalization of the composed density
# (pure numpy; the pathfinder is injected, no habitat import here)
from src.msp.navmesh_density import (
    DEFAULT_CELL_SIZE_M,
    DEFAULT_TAU_M,
    apply_density_masking,
    resolve_masking_mode,
)

# MAPG-02: real per-call accounting. One CallLog per episode; the
# runner's vlm_calls rollup is call_log.total(), retries included.
from src.results.calls import CallLog, model_name_of

# MAPG-10: compact scene-graph serialization for prompt text (stable
# prefix first, current agent pose only) and per-role model tiering.
from src.agents.serialization import (
    resolve_serialization_mode,
    serialize_scene_graph,
)

# Import the new Multi-Agent components. Agent classes are imported
# lazily inside __init__ per cfg agents_impl (MAPG-09): the unified
# stack (src/agents) needs no provider SDK at import time, while the
# legacy per-backend classes pull in their SDKs at module scope.
from src.multi_agent.blackboard import Blackboard

def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    """Read a possibly nested cfg value ('frames.gt_front') safely."""
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(part, None)
        else:
            node = getattr(node, part, None)
    return default if node is None else node


def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[MSP] Failed to write json {path}: {e}")

class MultiAgentMSPPlanner:
    def __init__(self, cfg, sg_sim, question, out_path=".", answer_mode="where", **kwargs):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self.out_path = Path(out_path)
        
        click.secho(f"\n{'='*40}\nINITIALIZING MULTI-AGENT PLANNER\n{'='*40}", fg="magenta", bold=True)
        click.secho(f"Question: {question}", fg="magenta")
        click.secho(f"Mode: {answer_mode}", fg="magenta")
        
        self.blackboard = Blackboard(question=question, mode=answer_mode)
        
        # Determine providers
        providers = kwargs.get("agent_providers", {})
        o_prov = providers.get("orchestrator", "claude")
        g_prov = providers.get("grounding", "claude")
        s_prov = providers.get("spatial", "claude")
        v_prov = providers.get("verifier", "claude")
        q_prov = providers.get("qa", "claude")

        click.secho(f"Providers: Orch={o_prov}, Ground={g_prov}, Spatial={s_prov}, Verif={v_prov}, QA={q_prov}", fg="yellow")

        # ------------------------------------------------------------------
        # MAPG-09: unified agent stack (src/agents): one prompt per
        # role shared across backends (byte-identical text,
        # golden-tested), typed outputs where a parse failure is a
        # counted schema_invalid, and adapters that return provider
        # usage so CallLog token counts are real. The legacy 24
        # per-backend agent files were deleted once the unified stack
        # reached parity; cfg agents_impl remains as an explicit guard
        # so stale legacy configs fail loudly instead of silently
        # running a different implementation than they name.
        # ------------------------------------------------------------------
        self.agents_impl = str(_cfg_get(self.cfg, "agents_impl", "unified")).lower().strip()
        if self.agents_impl == "legacy":
            raise RuntimeError(
                "agents_impl='legacy' is no longer available: the 24 "
                "per-backend agent files were deleted in MAPG-09 after "
                "the unified stack reached prompt and behavior parity "
                "(tests/test_golden_prompts.py). Use agents_impl='unified'."
            )
        if self.agents_impl != "unified":
            click.secho(
                f"[MSP] Unknown agents_impl={self.agents_impl!r}; using 'unified'.",
                fg="yellow",
            )
            self.agents_impl = "unified"
        click.secho(f"[MSP] agents_impl={self.agents_impl}", fg="cyan")

        # ------------------------------------------------------------------
        # MAPG-10: per-role model tiering. cfg model_tiers maps each
        # role to null (the provider's main model; default, no behavior
        # change) or a model-name override for that role only. The
        # backend adapter for a tiered role is constructed with the
        # override; CallLog records model_name per call so tiered runs
        # are attributable.
        # ------------------------------------------------------------------
        from src.agents.factory import create_role, resolve_model_tiers
        self.model_tiers, _tier_warnings = resolve_model_tiers(
            _cfg_get(self.cfg, "model_tiers", None)
        )
        for _w in _tier_warnings:
            click.secho(f"[MSP] {_w}", fg="yellow")
        _active_tiers = {r: m for r, m in self.model_tiers.items() if m}
        click.secho(
            f"[MSP] model_tiers={_active_tiers if _active_tiers else 'none (all roles on main model)'}",
            fg="cyan",
        )

        self.orchestrator = create_role(
            "orchestrator", provider=o_prov, model_name=self.model_tiers["orchestrator"]
        )
        self.grounder = create_role(
            "grounding", provider=g_prov, model_name=self.model_tiers["grounding"]
        )
        self.spatial = create_role(
            "spatial", provider=s_prov, model_name=self.model_tiers["spatial"]
        )
        self.verifier = create_role(
            "verifier", provider=v_prov, model_name=self.model_tiers["verifier"]
        )
        self.qa = create_role("qa", provider=q_prov, model_name=self.model_tiers["qa"])
        # Orphan role, deleted in MAPG-09 (constructed, never called).
        self.logical = None

        # ------------------------------------------------------------------
        # MAPG-10: scene-graph serialization mode for prompt text.
        # "compact" (default) = line format, 2 dp, stable prefix first,
        # current agent pose only (src/agents/serialization.py).
        # "legacy_json" = the historical full-precision node-link dump.
        # ------------------------------------------------------------------
        self.sg_serialization, _sg_warn = resolve_serialization_mode(
            _cfg_get(self.cfg, "sg_serialization", None)
        )
        if _sg_warn:
            click.secho(f"[MSP] {_sg_warn}", fg="yellow")
        click.secho(f"[MSP] sg_serialization={self.sg_serialization}", fg="cyan")

        # ------------------------------------------------------------------
        # MAPG-10 parse-once: the orchestrator decomposes a question
        # that never changes within an episode
        # (multi_agent_msp_planner used to re-call it every step). The
        # typed decomposition is memoized and only re-requested when
        # the failure history has changed since it was produced, which
        # preserves prompt rule 5 ("if your previous parsing failed,
        # choose a different interpretation"). d0 itself was already
        # parse-once (deterministic parse in __init__, a844d6b).
        # ------------------------------------------------------------------
        self._orch_out_cache = None
        self._orch_cache_history = None
        
        if "choices" in kwargs:
            self.blackboard.choices = kwargs["choices"]
        
        # Keep your existing robust math engine (numeric-conditioning aware:
        # question_dist may be None, in which case the metric kernel is
        # omitted from the composition instead of defaulting to 1.0 m).
        self.msp_engine = NumericConditioningEngine(
            sigma_s_factor=float(getattr(self.cfg, "sigma_s_factor", 0.5)),
            sigma_m_factor=float(getattr(self.cfg, "sigma_m_factor", 0.3)),
            kappa_factor=float(getattr(self.cfg, "kappa_factor", 10.0)),
        )

        # New configurable top_k parameter for returning best objects
        self.top_k = int(getattr(self.cfg, "top_k_objects", 2))

        # ------------------------------------------------------------------
        # MAPG-01: kernel bandwidths. cfg ``kernel_bandwidths`` is
        # "from_bbox" (default) or "fixed" (ablation arm, preserves the
        # historical hardcoded 0.5 m cube). Under from_bbox each
        # candidate's size comes from its Hydra AABB, and the engine is
        # handed an isotropic cube built from the anchor's max
        # horizontal extent s, so sigma_s = sigma_s_factor * s
        # (default 0.5 * s). Full formula set and frame conventions:
        # src.msp.kernel_bandwidths.py. A node without a valid box
        # falls back to the fixed cube with a recorded reason.
        # ------------------------------------------------------------------
        self.kernel_bandwidths, _kb_warn = resolve_bandwidth_mode(
            _cfg_get(self.cfg, "kernel_bandwidths", None)
        )
        if _kb_warn:
            click.secho(f"[MSP] {_kb_warn}", fg="yellow")
        click.secho(f"[MSP] kernel_bandwidths={self.kernel_bandwidths}", fg="cyan")
        # Per-candidate size provenance: {id: (source, reason)}.
        self._size_provenance = {}

        # ------------------------------------------------------------------
        # MAPG-12: navmesh masking + renormalization. cfg ``density_masking``
        # is "navmesh" (default: point estimation is the argmax navigable
        # cell of the masked, renormalized density) or "off" (ablation arm:
        # the pre-MAPG-12 unmasked estimator, bit-identical). When no
        # pathfinder is reachable through sg_sim the step falls back to off
        # with the reason recorded in pdf_params. The grid geometry is
        # navmesh-only, so it is built once and reused across steps.
        # ------------------------------------------------------------------
        self.density_masking, _dm_warn = resolve_masking_mode(
            _cfg_get(self.cfg, "density_masking", None)
        )
        if _dm_warn:
            click.secho(f"[MSP] {_dm_warn}", fg="yellow")
        self.navmesh_grid_m = float(
            _cfg_get(self.cfg, "navmesh_grid_m", DEFAULT_CELL_SIZE_M)
        )
        click.secho(
            f"[MSP] density_masking={self.density_masking} "
            f"navmesh_grid_m={self.navmesh_grid_m}",
            fg="cyan",
        )
        self._navmesh_grid = None

        # Persistent context
        self.locked_anchor_id = None

        # ------------------------------------------------------------------
        # P0 fix 1: d0 single source. Parse the metric literal ONCE per
        # episode with the deterministic parser. No default: value_m is
        # None when the utterance has no distance, and the metric kernel
        # is then omitted from the composition.
        # ------------------------------------------------------------------
        self.numeric_conditioning = str(
            _cfg_get(self.cfg, "numeric_conditioning", "on")
        ).lower().strip()
        if self.numeric_conditioning not in ("on", "off", "categorical_only"):
            click.secho(
                f"[MSP] Unknown numeric_conditioning={self.numeric_conditioning!r}; using 'on'.",
                fg="yellow",
            )
            self.numeric_conditioning = "on"
        self.metric_parse = parse_metric_literal(question)
        self.metric_schema_warnings = []
        click.secho(
            f"[MSP] Metric literal parse: value_m={self.metric_parse.value_m} "
            f"unit={self.metric_parse.unit} raw={self.metric_parse.raw!r} "
            f"warnings={self.metric_parse.warnings} "
            f"(numeric_conditioning={self.numeric_conditioning})",
            fg="cyan",
        )

        # ------------------------------------------------------------------
        # P0 fix 2: ann_yaw_rad plumbing. The runner passes the bench
        # annotation as anchor_front_yaw_world; it used to die in **kwargs.
        # It is GT, so its USE is gated behind cfg frames.gt_front (default
        # false, oracle-frames ablation only); by default it is only
        # recorded, never consumed.
        # ------------------------------------------------------------------
        _raw_yaw = kwargs.get("anchor_front_yaw_world", None)
        self.gt_anchor_front_yaw = float(_raw_yaw) if _raw_yaw is not None else None
        self.gt_front_available = self.gt_anchor_front_yaw is not None
        self.use_gt_front = bool(_cfg_get(self.cfg, "frames.gt_front", False))
        click.secho(
            f"[MSP] GT anchor front yaw: available={self.gt_front_available} "
            f"value={self.gt_anchor_front_yaw} frames.gt_front={self.use_gt_front}",
            fg="cyan",
        )

        # ------------------------------------------------------------------
        # P0 fix 3: verifier. LLM critique is optional (cfg verifier.llm,
        # default off per the ablation design); programmatic checks gate
        # every step. Verifier rejections retry via the existing
        # exploration path, capped at 2 retries per episode segment
        # (fairness rules); after the cap the failure is recorded and the
        # step proceeds instead of livelocking.
        # ------------------------------------------------------------------
        self.verifier_llm_enabled = bool(_cfg_get(self.cfg, "verifier.llm", False))
        try:
            self.verifier.llm_enabled = self.verifier_llm_enabled
        except Exception:
            pass
        self.verifier_rejections = 0
        self.max_verifier_retries = 2

        # ------------------------------------------------------------------
        # MAPG-02: per-call LLM accounting. Every agent invocation below
        # is wrapped with this log; the runner reads call_log.total()
        # for the episode vlm_calls rollup instead of assuming 4 calls
        # per step. Token counts are None until MAPG-09 makes the agent
        # classes return provider usage (they currently drop it).
        # ------------------------------------------------------------------
        self.call_log = CallLog()

    def _get_room_for_node(self, node_id: str) -> Optional[str]:
        """Traverse the Habitat scene graph hierarchy (Node -> Region -> Room)."""
        try:
            graph = getattr(self.sg_sim, "filtered_netx_graph", None)
            if not graph or not graph.has_node(node_id):
                return None
            
            # Walk up the edges to find the parent region or room
            for neighbor in graph.neighbors(node_id):
                neighbor_str = str(neighbor).lower()
                if "room" in neighbor_str:
                    return str(neighbor)
                if "region" in neighbor_str:
                    for room_candidate in graph.neighbors(neighbor):
                        if "room" in str(room_candidate).lower():
                            return str(room_candidate)
        except Exception:
            pass
        return None

    def _resolve_d0(self, orch_out: Dict[str, Any]):
        """Resolve d0 for this step. Returns (d0_m or None, provenance dict).

        d0 comes ONLY from the deterministic parser (parsed once per
        episode in __init__). The orchestrator's structured metric field,
        previously extracted and then discarded, is now consumed as a
        cross-check: if it is present and numeric it must agree with the
        deterministic parse; on disagreement a schema_warning is logged
        and the deterministic parse wins.
        """
        relation = ""
        orch_metric_raw = ""
        if isinstance(orch_out, dict):
            relation = str(orch_out.get("composition_logic", "") or "")
            for a in orch_out.get("anchors", []) or []:
                if isinstance(a, dict) and str(a.get("metric", "") or "").strip():
                    orch_metric_raw = str(a["metric"]).strip()
                    break
        relation_used = relation if relation and relation != "none" else (
            infer_relation(self.blackboard.question) or relation
        )

        mode = self.numeric_conditioning
        if mode == "off":
            return None, {
                "source": "numeric_conditioning_off",
                "relation": relation_used,
                "orchestrator_metric_raw": orch_metric_raw,
            }
        if mode == "categorical_only":
            d = resolve_categorical_distance(relation_used)
            return d, {
                "source": "categorical_default_table",
                "relation": relation_used,
                "orchestrator_metric_raw": orch_metric_raw,
            }

        # mode == "on": deterministic parse of the question is the value.
        d = self.metric_parse.value_m
        if orch_metric_raw:
            orch_parse = parse_metric_literal(orch_metric_raw)
            disagree = (
                orch_parse.value_m is None
                or d is None
                or abs(orch_parse.value_m - d) > 1e-6
            )
            if disagree:
                warning = (
                    f"schema_warning: orchestrator metric field {orch_metric_raw!r} "
                    f"(parsed {orch_parse.value_m}) disagrees with deterministic parse "
                    f"{d}; trusting the deterministic parse"
                )
                if warning not in self.metric_schema_warnings:
                    self.metric_schema_warnings.append(warning)
                    click.secho(f"[MSP] {warning}", fg="yellow")
        return d, {
            "source": "question_regex",
            "relation": relation_used,
            "orchestrator_metric_raw": orch_metric_raw,
        }

    def _scene_aabb(self, objects):
        """Scene AABB from the mapped object centroids (Habitat frame)."""
        if not objects:
            return None, None
        pts = np.asarray([o["position"] for o in objects], dtype=np.float32)
        return pts.min(axis=0).tolist(), pts.max(axis=0).tolist()

    def _get_pathfinder(self):
        """The habitat pathfinder reachable through sg_sim, or None.

        Same access pattern as the injected snap function: sg_sim may
        expose it directly or via its wrapped sim. MAPG-12 additionally
        uses it (get_bounds + snap_point) to build the navmesh grid.
        """
        try:
            pf = getattr(self.sg_sim, "pathfinder", None)
            if pf is None:
                sim = getattr(self.sg_sim, "sim", None)
                pf = getattr(sim, "pathfinder", None) if sim is not None else None
            return pf
        except Exception:
            return None

    def _get_navmesh_snap_fn(self):
        """Navmesh snap function, injectable and skippable.

        Outside the container (no habitat pathfinder reachable through
        sg_sim) this returns None and the on_navmesh check is recorded
        as skipped instead of silently passing.
        """
        try:
            pf = self._get_pathfinder()
            if pf is None or not hasattr(pf, "snap_point"):
                return None

            def _snap(p):
                try:
                    s = pf.snap_point([float(p[0]), float(p[1]), float(p[2])])
                    s = [float(s[0]), float(s[1]), float(s[2])]
                    if any(x != x for x in s):
                        return None
                    return s
                except Exception:
                    return None

            return _snap
        except Exception:
            return None

    def _resolve_candidate_size(self, oid):
        """MAPG-01: (size_hab, source, reason) for one object node.

        Under kernel_bandwidths=from_bbox the size is the node's real
        Hydra AABB extents reordered to the habitat y-up frame; a
        missing or invalid box falls back to the fixed 0.5 m cube with
        a recorded reason. Never raises.
        """
        extents = None
        extents_err = None
        if self.kernel_bandwidths == "from_bbox":
            getter = getattr(self.sg_sim, "get_extents_from_id", None)
            if getter is None:
                extents_err = "sg_sim has no get_extents_from_id"
            else:
                try:
                    extents = getter(oid)
                except Exception as e:
                    extents_err = f"extents lookup failed: {e}"
        if extents_err is not None:
            return list(FIXED_SIZE_M), "fixed_fallback", extents_err
        return resolve_object_size_hab(extents, self.kernel_bandwidths)

    def _get_scene_data(self):
        from src.envs.utils import pos_normal_to_habitat
        objects, frontiers = [], []
        self._size_provenance = {}
        for oid, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            pos_norm = self.sg_sim.get_position_from_id(oid)
            if pos_norm is not None:
                pos_hab = np.asarray(pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)), dtype=np.float32)
                # MAPG-01: real AABB extents (habitat y-up order) instead
                # of the historical hardcoded 0.5 m cube.
                size_hab, size_source, size_reason = self._resolve_candidate_size(oid)
                self._size_provenance[str(oid)] = (size_source, size_reason)
                if size_reason is not None:
                    click.secho(f"[MSP] size fallback for {oid}: {size_reason}", fg="yellow")
                objects.append({"id": str(oid), "name": str(name).lower(), "position": pos_hab.tolist(), "size": size_hab})

        for fid in getattr(self.sg_sim, "frontier_node_ids", []) or []:
            pos_norm = self.sg_sim.get_position_from_id(fid)
            if pos_norm is not None:
                pos_hab = np.asarray(pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)), dtype=np.float32)
                # Frontiers have no AABB; keep the fixed cube.
                frontiers.append({"id": str(fid), "name": "frontier", "position": pos_hab.tolist(), "size": list(FIXED_SIZE_M)})
        return objects, frontiers

    def _serialized_scene_graph(self) -> str:
        """Scene-graph prompt text under cfg sg_serialization (MAPG-10).

        compact: line format from the filtered netx graph (stable
        prefix first, current agent pose only). When the sim exposes no
        netx graph (stubs, legacy paths) or serialization fails, the
        legacy JSON string is used with the reason logged.
        """
        if self.sg_serialization == "compact":
            graph = getattr(self.sg_sim, "filtered_netx_graph", None)
            if graph is not None:
                try:
                    return serialize_scene_graph(
                        graph,
                        mode="compact",
                        current_agent_id=getattr(self.sg_sim, "curr_agent_id", None),
                    )
                except Exception as e:
                    click.secho(
                        f"[MSP] compact serialization failed ({e}); "
                        "falling back to legacy_json for this step.",
                        fg="yellow",
                    )
        return self.sg_sim.scene_graph_str

    def get_next_action(self, agent_yaw_rad: float = 0.0, agent_pos_hab: Optional[np.ndarray] = None):
        if agent_pos_hab is None:
            agent_pos_hab = np.array([0, 0, 0], dtype=np.float32)
            
        objects, frontiers = self._get_scene_data()
        img_path = get_latest_image(self.out_path)
        if img_path:
            img_path = str(img_path)
            
        agent_state_str = self.sg_sim.get_current_semantic_state_str()
        
        # --- Step Header Logging ---
        step_num = self.blackboard.step_t + 1

        # MAPG-02: a step that re-runs because the verifier rejected the
        # previous one (P0 fix 3 retry loop, bounded by
        # max_verifier_retries) is a retry; its calls are flagged so the
        # accounting can separate first-attempt from retry spend.
        is_retry_step = self.verifier_rejections > 0
        click.secho(f"\n{'='*20} MULTI-AGENT STEP {step_num} {'='*20}", fg="magenta", bold=True)
        click.secho(f"[Env] Pose: {agent_pos_hab.tolist()} | Yaw: {agent_yaw_rad:.3f} rad", fg="white")
        click.secho(f"[Env] Semantic State: {agent_state_str}", fg="white")
        click.secho(f"[Env] Found {len(objects)} Objects, {len(frontiers)} Frontiers", fg="white")
        # MAPG-10: what gets logged is what the LLM sees.
        sg_str = self._serialized_scene_graph()
        click.secho(f"[Scene Graph]\n{sg_str}", fg="blue")
        click.secho("-" * 60, fg="white")

        # 1. Update Blackboard
        self.blackboard.update_state(
            t=step_num,
            pose=agent_pos_hab,
            yaw=agent_yaw_rad,
            img_path=img_path,
            sg_str=sg_str,
            agent_state=agent_state_str,
            objects=objects,
            frontiers=frontiers
        )
        
        def finalize_step(target_pose, target_id, is_conf, conf, extra):
            """Helper to log trace and print final decision before returning."""
            click.secho(f"\n[DECISION] Action: {extra.get('action_type')} | Target ID: {extra.get('chosen_id')} | Conf: {conf:.2f}", fg="yellow", bold=True)
            if extra.get("thought"):
                click.secho(f"[DECISION] Thought: {extra.get('thought')}", fg="yellow")
            
            trace_dump = {
                "t": step_num,
                "agent_pose": agent_pos_hab.tolist(),
                "agent_yaw": agent_yaw_rad,
                "ledger": self.blackboard.event_ledger,
                "final_decision": extra
            }
            _write_json(self.out_path / f"trace_step_{step_num:03d}.json", trace_dump)
            return target_pose, target_id, is_conf, conf, extra

        # =====================================================================
        # MCQ FAST PATH OVERRIDE
        # =====================================================================
        if self.blackboard.choices:
            click.secho(f"[Planner] Multiple Choice Query detected. Executing QA Fast Path.", fg="cyan")
            qa_out = self.call_log.call(
                "qa", self.qa.process, self.blackboard,
                model_name=model_name_of(self.qa),
                is_retry=is_retry_step, step_idx=step_num,
            )
            if qa_out.get("ok", False):
                action_type = qa_out.get("action_type", "lookaround")
                chosen_id = qa_out.get("chosen_id", "NONE")
                ans = qa_out.get("answer", "")
                conf_val = qa_out.get("confidence", 0.0)
                
                # Resolve target pose
                target_pose = None
                if action_type in ["goto_object", "goto_frontier"] and chosen_id != "NONE":
                    target_pose = self.sg_sim.get_position_from_id(chosen_id)
                
                # If action is answer, we don't set a target
                if action_type == "answer":
                    target_id = ans
                    is_conf = (conf_val >= float(getattr(self.cfg, "pre_answer_conf_thresh", 0.8)))
                else:
                    target_id = chosen_id
                    is_conf = False
                    
                extra = {
                    "action_type": action_type,
                    "chosen_id": target_id,
                    "thought": qa_out.get("reasoning", "")
                }
                
                def fallback_step():
                    fid = str(frontiers[0]["id"]) if frontiers else ""
                    fallback_action = "goto_frontier" if fid else "lookaround"
                    return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": fallback_action, "chosen_id": fid, "thought": "QA Fast Path failed geometry. Fallback exploring."})
                
                if action_type in ["goto_object", "goto_frontier"] and target_pose is None:
                    return fallback_step()
                    
                return finalize_step(target_pose, target_id, is_conf, conf_val, extra)
            else:
                click.secho(f"[Planner] QA Fast Path crashed. Proceeding with standard fallback.", fg="red")
        
        # 2. Agent 1: Orchestrate (MAPG-10 parse-once: the question is
        # episode-constant, so the typed decomposition is memoized and
        # only re-requested after the failure history changes, i.e.
        # when prompt rule 5 could produce a different interpretation.
        # Failed/schema-invalid outputs are never cached.)
        if (
            self._orch_out_cache is not None
            and not self._orch_out_cache.get("error")
            and self._orch_cache_history == self.blackboard.global_history
        ):
            orch_out = self._orch_out_cache
            click.secho(
                "[MSP] Orchestrator parse reused from episode cache (parse-once).",
                fg="cyan",
            )
        else:
            orch_out = self.call_log.call(
                "orchestrator", self.orchestrator.process, self.blackboard,
                model_name=model_name_of(self.orchestrator),
                is_retry=is_retry_step, step_idx=step_num,
            )
            if isinstance(orch_out, dict) and not orch_out.get("error"):
                self._orch_out_cache = orch_out
                self._orch_cache_history = self.blackboard.global_history
            else:
                self._orch_out_cache = None
                self._orch_cache_history = None

        # 3. Agent 2: Ground
        ground_out = self.call_log.call(
            "grounding", self.grounder.process, self.blackboard, orch_out,
            model_name=model_name_of(self.grounder),
            is_retry=is_retry_step, step_idx=step_num,
        )
        


        if ground_out.get("needs_exploration", False) or not ground_out.get("grounded_anchors"):
            anchor_in_view = False
        else:
            anchor_in_view = ground_out["grounded_anchors"][0]["matched_object_id"] != "NONE"

        if not hasattr(self, "steps_since_anchor_seen"):
            self.steps_since_anchor_seen = 0

        if not self.locked_anchor_id:
            if not anchor_in_view:
                fid = str(frontiers[0]["id"]) if frontiers else ""
                action = "goto_frontier" if fid else "lookaround"
                return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": action, "chosen_id": fid, "thought": "Missing anchors. Exploring."})

            # Check primary anchor
            primary_anchor_id = ground_out["grounded_anchors"][0]["matched_object_id"]
            if primary_anchor_id == "NONE":
                 fid = str(frontiers[0]["id"]) if frontiers else ""
                 return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": "goto_frontier" if fid else "lookaround", "chosen_id": fid, "thought": "Anchor is NONE. Exploring."})
            
            # Lock the anchor once found
            self.locked_anchor_id = primary_anchor_id
            self.locked_anchor_pos = self.sg_sim.get_position_from_id(self.locked_anchor_id)
            self.steps_since_anchor_seen = 0
            click.secho(f"[Anchor Locked] ID: {self.locked_anchor_id}", fg="green", bold=True)
            # After locking, we navigate near it to verify
            return finalize_step(self.sg_sim.get_position_from_id(self.locked_anchor_id), self.locked_anchor_id, False, 0.0, {"action_type": "goto_object", "chosen_id": self.locked_anchor_id, "thought": "Anchor locked. Navigating to anchor to cross-reference visual with scene graph."})
            
        else:
            primary_anchor_id = self.locked_anchor_id
            if anchor_in_view and ground_out["grounded_anchors"][0]["matched_object_id"] == self.locked_anchor_id:
                self.steps_since_anchor_seen = 0
            else:
                self.steps_since_anchor_seen += 1
                
            if self.steps_since_anchor_seen > 5:
                # We haven't seen the anchor in 5 steps. Navigate back to it.
                self.steps_since_anchor_seen = 0
                return finalize_step(self.locked_anchor_pos, primary_anchor_id, False, 0.0, {"action_type": "goto_object", "chosen_id": primary_anchor_id, "thought": "Lost sight of anchor for 5 steps. Navigating back to its last known position."})
            
        primary_anchor_obj = next((o for o in objects if o["id"] == primary_anchor_id), objects[0])

        # 4. Agent 3: Spatial Geometry
        spatial_out = self.call_log.call(
            "spatial", self.spatial.process, self.blackboard, primary_anchor_obj,
            model_name=model_name_of(self.spatial),
            is_retry=is_retry_step, step_idx=step_num,
        )
        if not spatial_out.get("ok", False):
            return finalize_step(self.sg_sim.get_position_from_id(primary_anchor_id), primary_anchor_id, False, 0.0, {"action_type": "goto_object", "chosen_id": primary_anchor_id, "thought": "Spatial failed (likely occluded). Moving closer to object."})

        # (QA Agent call has been moved to the MCQ Fast Path above)

        # =====================================================================
        # 5. Run MSP Math (Probabilistic Scoring & Point Estimation)
        # =====================================================================
        # P0 fix 1: d0 comes ONLY from the deterministic parser (episode
        # parse in __init__, mode-resolved here). None means the metric
        # kernel is OMITTED from the composition, not defaulted to 1.0 m.
        dist_m, metric_provenance = self._resolve_d0(orch_out)
        metric_kernel_active = dist_m is not None
        anchor_pos = np.asarray(primary_anchor_obj["position"], dtype=np.float32)

        # P0 fix 2: GT anchor front yaw (bench ann_yaw_rad). Recorded
        # always; USED only under cfg frames.gt_front for intrinsic
        # relations (oracle-frames ablation). Default pipeline never
        # consumes it: it is ground truth.
        kernel_params_used = dict(spatial_out)
        gt_front_used = False
        if self.use_gt_front and self.gt_front_available:
            theta_override = gt_front_theta(
                self.gt_anchor_front_yaw, metric_provenance.get("relation")
            )
            if theta_override is not None:
                kernel_params_used["theta"] = float(theta_override)
                gt_front_used = True
                click.secho(
                    f"[MSP] frames.gt_front active: theta overridden to "
                    f"{theta_override:.4f} rad from GT anchor front yaw.",
                    fg="yellow",
                )

        # MAPG-01: bandwidths from the anchor's real AABB. The engine
        # derives sigma_s/sigma_m/kappa from max(anchor_size), so we
        # hand it an isotropic cube [s, s, s] with s the anchor's max
        # horizontal extent (habitat x/z; vertical excluded on purpose,
        # see src.msp.kernel_bandwidths.py):
        #   sigma_s = sigma_s_factor * s (default 0.5 * s).
        # Fixed mode and box fallbacks use the historical 0.5 m cube
        # (sigma_s = 0.25 m), preserving pre-MAPG-01 behavior.
        anchor_size_source, anchor_size_reason = self._size_provenance.get(
            str(primary_anchor_id), ("fixed_fallback", "anchor missing from size provenance")
        )
        anchor_size_hab = [float(v) for v in primary_anchor_obj.get("size", list(FIXED_SIZE_M))]
        engine_anchor_size, bandwidth_scale_m = bandwidth_size_hab(anchor_size_hab, anchor_size_source)

        msp_objects, msp_frontiers = self.msp_engine.score_candidates(
            objects=objects,
            frontiers=frontiers,
            anchor_pos_hab=anchor_pos,
            anchor_size=engine_anchor_size,
            kernel_params=kernel_params_used,
            question_dist=dist_m,
            planar=True,
            flatten_semantic=bool(getattr(self.cfg, "flatten_semantic", False))
        )

        point_estimate = self.msp_engine.estimate_point_from_pdf(
            anchor_pos_hab=anchor_pos,
            kernel_params=kernel_params_used,
            question_dist=dist_m,
            anchor_size=engine_anchor_size,
            planar=True,
            use_map=True
        )
        point_xyz = np.asarray(point_estimate["xyz_chosen_hab"], dtype=np.float32)

        # ------------------------------------------------------------------
        # MAPG-12: masked, renormalized point estimate. Under
        # density_masking=navmesh (default) the point is the argmax
        # navigable cell of the density masked to Omega_free and
        # renormalized (log_Z by logsumexp over navmesh grid cells), so
        # it is navigable by construction. The pdf params are exactly the
        # ones estimate_point_from_pdf scores with (anchor-only params,
        # flatten_semantic_for_where default True). Fallbacks (mode off,
        # no pathfinder, build failure) keep the unmasked estimate above
        # and record the reason.
        # ------------------------------------------------------------------
        masked_pdf_params = {
            k: v
            for k, v in self.msp_engine._build_anchor_only_params(
                anchor_pos_hab=anchor_pos,
                anchor_size=engine_anchor_size,
                distance_m=dist_m,
                kernel_params=kernel_params_used,
                planar=True,
                flatten_semantic=True,
            ).items()
            if not str(k).startswith("_")
        }
        masked_xyz, density_masking_record, self._navmesh_grid = apply_density_masking(
            mode=self.density_masking,
            pathfinder=self._get_pathfinder(),
            pdf_params=masked_pdf_params,
            cell_size_m=self.navmesh_grid_m,
            tau_m=DEFAULT_TAU_M,
            grid=self._navmesh_grid,
        )
        if masked_xyz is not None:
            point_xyz = np.asarray(masked_xyz, dtype=np.float32)
        elif density_masking_record.get("density_masking_reason"):
            click.secho(
                f"[MSP] density masking inactive: "
                f"{density_masking_record['density_masking_reason']}",
                fg="yellow",
            )

        # Extract the continuous PDF parameters from the shared debug trace
        # (All candidates share the same anchor-centric pdf params)
        extracted_pdf_params = {}
        if msp_objects:
            extracted_pdf_params = msp_objects[0].get("_msp_debug", {}).get("metric_semantic_params", {})
            predicates = msp_objects[0].get("_msp_debug", {}).get("predicate_params", {})
            extracted_pdf_params.update(predicates)
        # Ablation-visible provenance in pdf_params.
        extracted_pdf_params["metric_kernel_active"] = bool(metric_kernel_active)
        extracted_pdf_params["gt_anchor_front_yaw"] = self.gt_anchor_front_yaw
        extracted_pdf_params["gt_front_used"] = bool(gt_front_used)
        # MAPG-01: bandwidth size provenance (sigma_s = sigma_s_factor *
        # bandwidth_scale_m; scale is the anchor's max horizontal extent
        # under from_bbox, 0.5 m under fixed or any fallback).
        extracted_pdf_params["kernel_bandwidths"] = self.kernel_bandwidths
        extracted_pdf_params["anchor_size_source"] = anchor_size_source
        extracted_pdf_params["anchor_size_fallback_reason"] = anchor_size_reason
        extracted_pdf_params["anchor_size_hab"] = anchor_size_hab
        extracted_pdf_params["bandwidth_scale_m"] = float(bandwidth_scale_m)
        # MAPG-12: masking provenance (density_masked, log_Z,
        # mass_in_tau_ball at tau = 1.0 m, fallback reason if any).
        extracted_pdf_params.update(density_masking_record)

        metric_parse_record = {
            "value_m": self.metric_parse.value_m,
            "unit": self.metric_parse.unit,
            "raw": self.metric_parse.raw,
            "warnings": list(self.metric_parse.warnings),
            "schema_warnings": list(self.metric_schema_warnings),
            "numeric_conditioning": self.numeric_conditioning,
            "d0_used_m": dist_m,
            "provenance": metric_provenance,
        }
        gt_front_record = {
            "available": bool(self.gt_front_available),
            "value_rad": self.gt_anchor_front_yaw,
            "cfg_enabled": bool(self.use_gt_front),
            "used": bool(gt_front_used),
        }

        # =====================================================================
        # 6. Verify (P0 fix 3: programmatic checks first, LLM optional)
        # =====================================================================
        scene_min, scene_max = self._scene_aabb(objects)
        checks = run_checks(
            spatial_payload=spatial_out,
            required_fields=("theta", "phi"),
            prediction_xyz=point_xyz.tolist(),
            anchor_xyz=anchor_pos.tolist(),
            d0_m=dist_m,
            sigma_m=(float(point_estimate.get("sigma_m_used")) if metric_kernel_active else None),
            scene_min=scene_min,
            scene_max=scene_max,
            navmesh_snap_fn=self._get_navmesh_snap_fn(),
        )
        # MAPG-02: the verifier only hits the LLM when the critique is
        # enabled AND the programmatic checks passed; it reports that via
        # llm_used. record_if keeps programmatic-only verifications out
        # of the call count.
        verification = self.call_log.call(
            "verifier", self.verifier.process, self.blackboard, checks=checks,
            model_name=model_name_of(self.verifier),
            is_retry=is_retry_step, step_idx=step_num,
            record_if=lambda out: bool(isinstance(out, dict) and out.get("llm_used", False)),
        )
        verifier_record = {
            "status": verification.get("status"),
            "feedback": verification.get("feedback", ""),
            "llm_used": verification.get("llm_used", False),
            "llm_error": verification.get("llm_error"),
            "checks": checks,
            "rejections_so_far": self.verifier_rejections,
        }

        if verification.get("status") == "FAIL":
            if self.verifier_rejections < self.max_verifier_retries:
                # Existing retry path: feedback into global history, one
                # exploration step, bounded at max_verifier_retries.
                self.verifier_rejections += 1
                verifier_record["rejections_so_far"] = self.verifier_rejections
                self.blackboard.global_history += (
                    f"Step {step_num} FAIL: {verification.get('feedback')}\n"
                )
                fid = str(frontiers[0]["id"]) if frontiers else ""
                return finalize_step(
                    self.sg_sim.get_position_from_id(fid) if fid else None,
                    fid, False, 0.0,
                    {
                        "action_type": "lookaround",
                        "chosen_id": "",
                        "thought": f"Verifier rejected logic: {verification.get('feedback')}",
                        "verifier": verifier_record,
                        "metric_parse": metric_parse_record,
                        "metric_kernel_active": bool(metric_kernel_active),
                        "gt_front": gt_front_record,
                    },
                )
            # Retry budget exhausted: record the terminal failure and
            # proceed instead of livelocking on exploration.
            verifier_record["retry_budget_exhausted"] = True
            self.blackboard.global_history += (
                f"Step {step_num} verifier FAIL after {self.max_verifier_retries} "
                f"retries; proceeding with best-effort answer.\n"
            )
            click.secho(
                "[Verifier] FAIL after retry budget exhausted; proceeding.",
                fg="red",
            )
        else:
            self.verifier_rejections = 0

        # Build output structure with PDF, point, and top K objects
        
        # --- NEW: Top-Down 2D Matplotlib Heatmap Export ---
        if extracted_pdf_params:
            try:
                import matplotlib.pyplot as plt
                from src.msp.pdf import combined_logpdf as _combined_logpdf
                
                # Create a 10x10m grid centered around the anchor object
                grid_res = 0.2
                x_min, x_max = anchor_pos[0] - 5.0, anchor_pos[0] + 5.0
                z_min, z_max = anchor_pos[2] - 5.0, anchor_pos[2] + 5.0
                
                xx, zz = np.meshgrid(
                    np.arange(x_min, x_max, grid_res),
                    np.arange(z_min, z_max, grid_res)
                )
                yy = np.full_like(xx, anchor_pos[1]) # Keep Y (elevation) constant
                
                # Flatten the grid for logpdf evaluation
                grid_logps = _combined_logpdf(xx.ravel(), yy.ravel(), zz.ravel(), extracted_pdf_params)
                grid_logps = grid_logps.reshape(xx.shape)
                
                # Plot the heat map
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # We use origin='lower' to match array indexing with Cartesian Y-up (which is Z visually in Habitat topdown)
                c = ax.pcolormesh(xx, zz, np.exp(grid_logps), shading='auto', cmap='viridis', alpha=0.8)
                plt.colorbar(c, ax=ax, label='Probability Density')
                
                ax.plot(anchor_pos[0], anchor_pos[2], 'r*', markersize=15, label='Anchor Object')
                ax.plot(agent_pos_hab[0], agent_pos_hab[2], 'b^', markersize=12, label='Agent')
                ax.plot(point_xyz[0], point_xyz[2], 'gx', markersize=12, label='MSP Target Guess')
                
                ax.set_title(f"Step {step_num} Point-Estimation Heatmap")
                ax.set_xlabel("X (World)")
                ax.set_ylabel("Z (World) [Top-down Depth]")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                heatmap_file = self.out_path / f"heatmap_step_{step_num:03d}.png"
                plt.savefig(heatmap_file, dpi=150)
                plt.close(fig)
                click.secho(f"[MSP] Exported 2D heatmap to {heatmap_file.name}", fg="cyan")
            except Exception as e:
                click.secho(f"[MSP] Failed to generate 2D heatmap: {e}", fg="red")

        top_k_objects = []
        top_k_objects = []
        for obj in msp_objects[:self.top_k]:
             top_k_objects.append({
                 "id": obj["id"],
                 "name": obj.get("name", ""),
                 "position": obj["position"],
                 "confidence": float(np.exp(obj.get("msp_score", -100.0))) # Convert logpdf roughly to score
             })

        # =====================================================================
        # 7. Final Action Decision (Topological Bounding + PDF Ranking)
        # =====================================================================
        anchor_room_id = self._get_room_for_node(primary_anchor_id)
        
        same_room_objects = []
        same_room_frontiers = []
        
        for obj in msp_objects:
            if obj["id"] == primary_anchor_id:
                continue
            obj_room_id = self._get_room_for_node(obj["id"])
            if anchor_room_id and obj_room_id:
                if anchor_room_id == obj_room_id:
                    same_room_objects.append(obj)
            elif np.linalg.norm(np.array(obj["position"]) - anchor_pos) < 6.0:
                same_room_objects.append(obj)

        for f in msp_frontiers:
            f_room_id = self._get_room_for_node(f["id"])
            if anchor_room_id and f_room_id:
                if anchor_room_id == f_room_id:
                    same_room_frontiers.append(f)
            elif np.linalg.norm(np.array(f["position"]) - anchor_pos) < 6.0:
                same_room_frontiers.append(f)

        room_fully_explored = (len(same_room_frontiers) == 0)
        global_map_exhausted = (len(frontiers) == 0)
        
        # Format the comprehensive output
        def build_answer(action_type, chosen_id, conf_score, target_hab, thought):
             # normalize_prediction keeps target_location for backward
             # compatibility and adds the canonical target_point_xyz key.
             return finalize_step(target_hab, chosen_id, True if conf_score > 0.9 else False, conf_score, normalize_prediction({
                 "action_type": action_type,
                 "chosen_id": chosen_id,
                 "confidence": conf_score,
                 "thought": thought,
                 "pdf_params": extracted_pdf_params,
                 "target_location": point_xyz.tolist(),
                 "top_k_objects": top_k_objects,
                 "metric_parse": metric_parse_record,
                 "metric_kernel_active": bool(metric_kernel_active),
                 "gt_front": gt_front_record,
                 "verifier": verifier_record
             }))
             
        is_location_target = any(w in orch_out.get("target_entity", "").lower() for w in ["location", "region", "point", "place", "area"])

        # LOGIC A: If the target is explicitly a location, and we calculated the geometry successfully, go there directly!
        if is_location_target and self.locked_anchor_id:
             return build_answer("goto_object", "POINT_GUESS", 0.95, point_xyz, "Anchor is locked and spatial geometry is calculated. Navigating directly to the requested continuous location.")

        # LOGIC B: If the room is fully explored, trust the PDF rankings.
        if room_fully_explored or global_map_exhausted:
            if same_room_objects:
                best_obj = same_room_objects[0]
                conf_score = 0.95 
                thought = f"Room {anchor_room_id} fully explored. Selected highest PDF-scoring object."
                return build_answer("answer", best_obj["id"], conf_score, None, thought)
            else:
                return build_answer("goto_object", "POINT_GUESS", 0.0, point_xyz, "Room explored but target missing. Navigating to raw PDF median target location.")

        # LOGIC C: Room is NOT fully explored, but we found a fantastic match early.
        if same_room_objects and float(same_room_objects[0].get("msp_score", -100.0)) > -1.5:
             best_obj = same_room_objects[0]
             return build_answer("answer", best_obj["id"], 0.95, None, f"Extremely high PDF match found.")

        # LOGIC D: Keep exploring the remaining frontiers in this room.
        if same_room_frontiers:
            fid = same_room_frontiers[0]["id"]
            return finalize_step(self.sg_sim.get_position_from_id(fid), fid, False, 0.0, {
                "action_type": "goto_frontier",
                "chosen_id": fid,
                "thought": f"Exploring remaining frontiers inside the anchor's room ({anchor_room_id}).",
                "verifier": verifier_record,
                "metric_parse": metric_parse_record,
                "metric_kernel_active": bool(metric_kernel_active),
                "gt_front": gt_front_record
            })

        # Fallback
        fid = str(frontiers[0]["id"]) if frontiers else ""
        return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {
            "action_type": "lookaround" if not fid else "goto_frontier",
            "chosen_id": fid,
            "thought": "Fallback exploration triggered.",
            "verifier": verifier_record,
            "metric_parse": metric_parse_record,
            "metric_kernel_active": bool(metric_kernel_active),
            "gt_front": gt_front_record
        })
