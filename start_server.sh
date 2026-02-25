#!/bin/bash

# CHANGE MODEL AND DUMP FOLDERS to your own.
MODEL_FOLDER="/depot/cms/users/colberte/SONIC/sonic-inference-replay/single_sonic_model/models"
DUMP_FOLDER="/depot/cms/users/colberte/SONIC/sonic-inference-replay/replay_dumps"

TRITON_CONTAINER="/depot/cms/users/colberte/SONIC/triton_25.07.sif"
MY_PACKAGES="/depot/cms/private/users/colberte/conda_envs/torchonnxCUDA/lib/python3.12/site-packages"
CONTAINER_PACKAGES="/usr/local/lib/python3.12/dist-packages"

# Set to 1 to dump every request's inputs to /dumps as .npz (for testing/debugging)
SONIC_DUMP_INPUTS="${SONIC_DUMP_INPUTS:-0}"

apptainer run --nv \
    --env SONIC_DUMP_INPUTS="${SONIC_DUMP_INPUTS}" \
    -B "${MODEL_FOLDER}:/models" -B "${DUMP_FOLDER}:/dumps" \
    -B "${MY_PACKAGES}/functorch:${CONTAINER_PACKAGES}/functorch" \
    -B "${MY_PACKAGES}/torch:${CONTAINER_PACKAGES}/torch" \
    -B "${MY_PACKAGES}/torch-2.6.0+cu126.dist-info:${CONTAINER_PACKAGES}/torch-2.6.0+cu126.dist-info" \
    -B "${MY_PACKAGES}/torchgen:${CONTAINER_PACKAGES}/torchgen" \
    $TRITON_CONTAINER tritonserver --model-repository=/models
