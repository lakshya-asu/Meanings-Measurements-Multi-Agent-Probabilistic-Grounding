#!/bin/bash

# Name of the Docker container
CONTAINER_NAME="mapg_dev"

# Docker image to use
DOCKER_IMAGE="flux04/mapg_habitat:src"

# Path to the workspace directory
WORKSPACE_DIR="$(pwd)"

# Run the Docker container with the appropriate arguments
docker run -it \
  --name $CONTAINER_NAME \
  --privileged \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
  --env="XAUTHORITY:$XAUTHORITY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e QT_X11_NO_MITSHM=1 \
  -v $WORKSPACE_DIR:/workspace \
  -v $WORKSPACE_DIR/datasets:/datasets \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --rm \
  $DOCKER_IMAGE \
  /bin/bash
