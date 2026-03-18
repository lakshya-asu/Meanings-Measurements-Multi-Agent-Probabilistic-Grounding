import click
import hydra_python as hydra
from pathlib import Path
from omegaconf import OmegaConf

from hydra_python._hydra_bindings import PythonConfig
import pathlib
import click
import yaml


DEFAULT_CONFIGS = [
    ("frontend", "frontend_config.yaml"),
    ("backend", "backend_config.yaml"),
    ("reconstruction", "reconstruction_config.yaml"),
]


def _label_space_dirs() -> list[Path]:
    """Return label-space directories in priority order."""
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "src" / "cfg" / "label_spaces",
        repo_root / "cfg" / "label_spaces",
        # Legacy fallback for older checkouts.
        repo_root / "config" / "label_spaces",
    ]


def _resolve_label_space_path(labelspace_name: str) -> Path | None:
    filename = f"{labelspace_name}_label_space.yaml"
    for base_dir in _label_space_dirs():
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return None

def load_configs(
    dataset_name: str,
    labelspace_name: str = "ade20k_mp3d",
    bounding_box_type: str = "AABB",
):
    """
    Load various configs to construct the Hydra pipeline.

    dataset_name: Dataset name to load config from
    labelspace_name: Labelspace name to use
    bounding_box_type: Type of bounding box to use

    Returns:
        (Optional[PythonConfig]) Pipline config or none if invalid
    """
    hydra_config_path = hydra.get_config_path()

    dataset_path = hydra_config_path / dataset_name
    if not dataset_path.exists():
        click.secho(f"invalid dataset path: {dataset_path}", fg="red")
        return None

    labelspace_path = _resolve_label_space_path(labelspace_name)
    if labelspace_path is None:
        searched_dirs = ", ".join(str(p) for p in _label_space_dirs())
        click.secho(
            f"invalid labelspace '{labelspace_name}'. searched: {searched_dirs}",
            fg="red",
        )
        return None

    configs = PythonConfig()
    for ns, config_name in DEFAULT_CONFIGS:
        dataset_config = dataset_path / config_name
        if dataset_config.exists():
            configs.add_file(dataset_config, config_ns=ns)

    configs.add_file(labelspace_path)

    pipeline = {
        "frontend": {"type": "FrontendModule"},
        "backend": {"type": "BackendModule"},
        "reconstruction": {"type": "ReconstructionModule"},
    }
    configs.add_yaml(yaml.dump(pipeline))

    overrides = {
        "frontend": {"objects": {"bounding_box_type": bounding_box_type}},
        "lcd": {"lcd_use_bow_vectors": False},
        "reconstruction": {
            "show_stats": False,
            "pose_graphs": {"make_pose_graph": True},
        },
    }
    configs.add_yaml(yaml.dump(overrides))
    return configs

def initialize_hydra_pipeline(cfg, habitat_data, output_path):
    hydra.set_glog_level(cfg.glog_level, cfg.verbosity)
    configs = load_configs("habitat", labelspace_name=cfg.label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{cfg.label_space}'",
            fg="red",
        )
        return
    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True

    if habitat_data.cfg.use_semantic_data:
        pipeline_config.label_names = {i: x for i, x in enumerate(habitat_data.colormap.names)}
        habitat_data.colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    else:
        config_path = _resolve_label_space_path("hm3d")
        if config_path is None:
            click.secho("Missing hm3d label space yaml", fg="red")
            return None
        hm3d_labelspace = OmegaConf.load(config_path)
        names = [d.name for d in hm3d_labelspace.label_names]
        colormap = hydra.SegmentationColormap.from_names(names=names)
        pipeline_config.label_names = {i: x for i, x in enumerate(colormap.names)}
        colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)
    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(habitat_data.camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline


def initialize_hydra_pipeline_stretch(cfg, obs, output_path, sensor_categories_mapping=None):
    from src.stretch_ai_agent.utils import write_config_yaml

    write_config_yaml(sensor_categories_mapping, label_space=cfg.label_space)
    # Get camera info
    camera_K = obs.camera_K
    width = obs.rgb.shape[1]
    height = obs.rgb.shape[0]
    camera_info = {
        "fx": float(camera_K[0,0]),
        "fy": float(camera_K[1,1]),
        "cx": float(camera_K[0,2]),
        "cy": float(camera_K[1,2]),
        "width": width,
        "height": height,
    }

    hydra.set_glog_level(cfg.glog_level, cfg.verbosity)
    configs = load_configs("habitat", labelspace_name=cfg.label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{cfg.label_space}'",
            fg="red",
        )
        return
    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True

    if sensor_categories_mapping is not None:
        names = [v for k, v in sensor_categories_mapping.items()]
    else:
        click.secho(f"Using default label space from habitat'",fg="red",)
        config_path = _resolve_label_space_path("hm3d")
        if config_path is None:
            click.secho("Missing hm3d label space yaml", fg="red")
            return None
        hm3d_labelspace = OmegaConf.load(config_path)
        names = [d.name for d in hm3d_labelspace.label_names]

    colormap = hydra.SegmentationColormap.from_names(names=names)
    pipeline_config.label_names = {i: x for i, x in enumerate(colormap.names)}
    colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)
    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline
