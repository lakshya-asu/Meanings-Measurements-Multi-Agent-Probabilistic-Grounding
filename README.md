# MAPG: Meanings and Measurements

<p align="center">
  <b>Multi-Agent Probabilistic Grounding for Vision-Language Navigation</b>
</p>

<p align="center">
  <a href="https://youtu.be/zK7Ya-g9eXg">
    <img src="https://img.shields.io/badge/Demo-YouTube-red?style=for-the-badge&logo=youtube" alt="YouTube Demo" />
  </a>
  <a href="./Meanings%20and%20Measurements.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-blue?style=for-the-badge&logo=adobeacrobatreader" alt="Paper PDF" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv placeholder" />
  </a>
  <a href="https://hub.docker.com/r/flux04/mapg_habitat">
    <img src="https://img.shields.io/badge/Docker-flux04%2Fmapg__habitat-2496ED?style=for-the-badge&logo=docker" alt="Docker image" />
  </a>
  <a href="./README_SETUP.md">
    <img src="https://img.shields.io/badge/Setup-Guide-2ea44f?style=for-the-badge" alt="Setup Guide" />
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/zK7Ya-g9eXg">
    <img src="https://img.youtube.com/vi/zK7Ya-g9eXg/maxresdefault.jpg" alt="MAPG demo video" width="900" />
  </a>
</p>

<p align="center">
  Click the preview to watch the MAPG demo on YouTube, or open <a href="./MAPG_video.mp4"><code>MAPG_video.mp4</code></a> directly from the repository.
</p>

This repository contains the code for **MAPG** (Multi-Agent Probabilistic Grounding), introduced in the paper **"Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation."**

MAPG addresses a core challenge in embodied language grounding: natural language instructions often combine **semantic references**, **spatial relations**, and **metric constraints**. A command such as "go two meters to the right of the fridge" is not just object recognition, and not just navigation. It requires resolving a referent, grounding a spatial predicate in 3D, and producing an actionable goal in continuous space. MAPG tackles this by decomposing language into structured components, grounding them with specialized agents, and composing the results probabilistically into planner-ready target distributions.

The repository includes the current MAPG Habitat workflow, benchmarking code, multi-agent reasoning components, ablations, and a real-world Robotis entrypoint.

## Paper

- Paper PDF: [Meanings and Measurements.pdf](./Meanings%20and%20Measurements.pdf)
- arXiv: coming soon
- Project page: from the paper, the accompanying webpage is listed as `sites.google.com/view/mapg-web/home`

## What This Repo Contains

- **MAPG planners** for metric-semantic grounding and multi-agent reasoning
- **Habitat-based benchmark runners** for simulation experiments
- **MAPG-Bench / HM-EQA style evaluation code and configs**
- **Ablation scripts** for controlled comparisons
- **Robotis real-world entrypoint** for deployment with structured scene information
- **Dockerized setup** for collaborators who need the full environment reproducibly

## Method Overview

MAPG is built around a modular, agentic pipeline:

1. **Orchestrator** parses a free-form instruction into structured pieces such as anchor object, spatial predicate, and metric constraint.
2. **Grounding agents** resolve candidate referents using scene-graph context and visual evidence.
3. **Spatial reasoning modules** convert metric and relational constraints into analytic likelihoods in 3D space.
4. **Probabilistic composition** combines those constraints into a continuous goal distribution.
5. **Planner execution** uses that distribution to choose actionable targets in the scene.

This design is meant to bridge the gap between high-level VLM reasoning and physically grounded navigation in continuous 3D environments.

## Main Entry Points

The current repo is centered around a small set of active scripts:

- Multi-agent benchmark:
  `python src/scripts/run_multi_agent_benchmark.py -cf spatial_vqa`
- Single-agent MSP benchmark:
  `python src/scripts/run_msp_benchmark.py -cf spatial_vqa`
- MSP ablation:
  `python src/scripts/msp_ablation.py -cf ablation_run`
- Robotis / real-world run:
  `python src/scripts/run_robotis_qwen.py`

## Repository Layout

```text
src/
├── cfg/                 # experiment configs
├── envs/                # Habitat and environment interfaces
├── logging/             # logging utilities
├── msp/                 # metric-semantic probabilistic utilities
├── multi_agent/         # multi-agent reasoning stack
├── occupancy_mapping/   # scene and map utilities
├── planners/            # active MAPG planners
├── real_world/          # Robotis / deployment-facing code
├── scene_graph/         # scene graph integration
├── scripts/             # runnable experiment entrypoints
├── tools/               # inspection and helper tools
└── utils/               # shared utilities
```

## Setup

For the reproducible Docker-based setup, dataset download instructions, image usage, and collaborator workflow, see:

- [README_SETUP.md](./README_SETUP.md)

That guide covers:

- pulling the Docker image
- downloading the dataset bundle
- launching the container with Docker Compose
- running the main MAPG scripts

## Docker

The published Habitat image is currently named:

```bash
flux04/mapg_habitat:src
```

To build it locally:

```bash
docker build -f Dockerfile_habitat -t flux04/mapg_habitat:src .
```

To run with Compose:

```bash
docker compose up -d
docker compose exec grapheqa bash
```

## Datasets

The setup guide expects the dataset bundle to be placed under:

```text
datasets/
```

The shared download location currently used for this project is:

- `https://drive.google.com/drive/folders/1YiWecgga3Eh7GWsdlQEv2iaklzO6MaQ7?usp=drive_link`

## Status

This repo reflects the current MAPG codebase used for Habitat benchmarking and the associated real-world demo path. Older experimental branches and legacy GraphEQA components have been pared back, and the active workflow is now centered on the MAPG planners and scripts under [`src/`](./src).

## Citation

Citation text will be added once the arXiv version and final citation block are ready.

For now, if you reference this repository, please cite the paper title:

```text
Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation
```

## Contact

If you are using this repository for research, reproducing results, or integrating MAPG into your own stack, the best starting point is the setup guide plus the main benchmark scripts listed above.

Current repository maintainer:

- `lakshya-asu`
