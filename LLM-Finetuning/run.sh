#!/bin/bash
cat<<'EOT'
  ____                ____       _____              _     
 |  _ \ _   _ _ __   |  _ \ _   |_   _|__  _ __ ___| |__  
 | |_) | | | | '_ \  | |_) | | | || |/ _ \| '__/ __| '_ \ 
 |  _ <| |_| | | | | |  __/| |_| || | (_) | | | (__| | | |
 |_| \_\\__,_|_| |_| |_|    \__, ||_|\___/|_|  \___|_| |_|
                            |___/                         
EOT
# This script is used to build the project and run the tests.

# determine whether podman or docker should be used
if [ -x "$(command -v podman)" ]; then
    DOCKER=podman
    DEVICES="--device nvidia.com/gpu=0 --device nvidia.com/gpu=1 --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 --security-opt=label=disable"
elif [ -x "$(command -v docker)" ]; then
    DOCKER=docker
    DEVICES="--device /dev/dri:/dev/dri"
else
    echo "Neither podman nor docker is installed. Please install one of them."
    exit 1
fi

CONTAINER_NAME="${USER}-nvidia-ml-pytorch"

HUGGINGFACE_TOKEN=$(cat $HOME/.secrets/huggingface_token.txt)

# $DOCKER build -t $CONTAINER_NAME \
#     --build-arg NVIDIA_VISIBLE_DEVICES=all \
#     -f Dockerfile .

echo "Running container $CONTAINER_NAME"
$DOCKER run --rm -it \
    $DEVICES \
    --name $CONTAINER_NAME \
    --workdir /app \
    $CONTAINER_NAME $*

#     -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \



