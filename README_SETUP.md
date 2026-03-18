# MAPG Setup Guide

This guide is for collaborators who need to run the current Habitat workflow from:

- this GitHub repo
- the shared dataset bundle
- the Docker image we publish

It assumes Linux with Docker, Docker Compose, and an NVIDIA GPU.

## What collaborators need

1. Clone the repo:
   `https://github.com/TechTinkerPradhan/MAPG`
2. Download the dataset bundles:

### Download the HM3D dataset
The HM3D dataset along with semantic annotations can be downloaded from [here](https://github.com/matterport/habitat-matterport-3dresearch). Follow the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) to download together the train/val scenes, semantic annotations and configs. The data folder should look as follows:

```graphql
hm3d/train
├─ hm3d-train-semantic-annots-v0.2
|    ├─ ...
├─ 00366-fxbzYAGkrtm
|    ├─ fxbzYAGkrtm.basis.glb
|    ├─ fxbzYAGkrtm.basis.navmesh
|    ├─ fxbzYAGkrtm.semantic.glb
|    ├─ fxbzYAGkrtm.semantic.txt
├─ ...
```

### Download Explore-EQA dataset
Navigate to [this link](https://drive.google.com/drive/folders/1YiWecgga3Eh7GWsdlQEv2iaklzO6MaQ7?usp=drive_link) and download `msp_questions.csv` and `scene_init_poses_semantics_only.csv` into explore-eqa in datasets/ directory in your workspace.  

Finally datasets/  should look like :-
datasets/  
   ├─ explore-eqa
      ├─ msp_questions.csv
      ├─ scene_init_poses_semantics_only.csv
   ├─ hm3d/train
      ├─ hm3d-train-semantic-annots-v0.2
      |    ├─ ...
      ├─ 00366-fxbzYAGkrtm
      |    ├─ fxbzYAGkrtm.basis.glb
      |    ├─ fxbzYAGkrtm.basis.navmesh
      |    ├─ fxbzYAGkrtm.semantic.glb
      |    ├─ fxbzYAGkrtm.semantic.txt
      ├─ ...
3. Pull the Docker image we publish:
   `flux04/mapg_habitat:src`

## Expected host layout

```text
MAPG/
├── datasets/
├── docker-compose.yml
├── .env
├── src/
└── ...
```

The container expects:

- repo mounted at `/workspace`
- datasets mounted at `/datasets`

## 1. Clone the repo

```bash
cd ~
git clone https://github.com/TechTinkerPradhan/MAPG
cd MAPG
```

## 2. Download the datasets

Download the Google Drive folder and place its contents under:

```bash
~/MAPG/datasets
```

Quick sanity check:

```bash
ls -lah datasets
```

## 3. Create the env file

Copy the example file:

```bash
cp .env.example .env
```

Edit `.env` and fill in the keys you actually use:

```bash
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
GRAPH_EQA_IMAGE=flux04/mapg_habitat:src
```

## 4. Pull the published image

```bash
docker pull flux04/mapg_habitat:src
```

## 5. Start the container

From the repo root:

```bash
docker compose up -d
```

Open a shell:

```bash
docker compose exec mapg bash
```

Inside the container, confirm the environment:

```bash
conda activate mapg
cd /workspace
nvidia-smi
ls -lah /datasets | head
```

## 6. Run the current benchmark scripts

Main single-agent MSP benchmark:

```bash
python src/scripts/run_msp_benchmark.py -cf spatial_vqa
```

Main multi-agent MSP benchmark:

```bash
python src/scripts/run_multi_agent_benchmark.py -cf spatial_vqa
```

Robotis / real-world script:

```bash
python src/scripts/run_robotis_qwen.py
```

MSP ablation:

```bash
python src/scripts/msp_ablation.py -cf ablation_run
```

## 7. Build the image locally

If you are the maintainer and want to rebuild the image from this repo:

```bash
docker build -f Dockerfile_habitat -t flux04/mapg_habitat:src .
```

Then optionally push it:

```bash
docker push flux04/mapg_habitat:src
```

## 8. Notes for collaborators

- The Docker image already contains the heavy software stack.
- Collaborators still need the repo because scripts/configs are mounted from the checkout.
- Collaborators still need the dataset bundle because it is mounted from the host into `/datasets`.
- If a script expects an API key, that key must be present in `.env`.

## 9. Useful commands

Start:

```bash
docker compose up -d
```

Shell:

```bash
docker compose exec mapg bash
```

Logs:

```bash
docker compose logs -f --tail=200
```

Stop:

```bash
docker compose down
```
