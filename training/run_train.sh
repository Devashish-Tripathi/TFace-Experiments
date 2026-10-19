#!/bin/bash
# other details can be changed from the yaml file

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export OMP_NUM_THREADS=1
LOG_DIR="/home/devashish_tripathi/projects/sanity_logs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/training$(date +%Y%m%d_%H%M%S).log"
# LOG_FILE="${LOG_DIR}/training_nonoise$(date +%Y%m%d_%H%M%S).log"

YAML_NAME="train_sanity.yaml"
EPSILON=0.5
DELTA=1e-5

# TOGGLES
USE_NO_NOISE=false # keep false to mimic baseline code. true means no noise is added
USE_NO_LOCS=true # keep false to mimic baseline code. true means no location is used
USE_SENSITIVITY=true # keep false to mimic baseline code. true means sensitivity is used
USE_GAUSSIAN=true # keep false to mimic baseline code. true means use Gaussian mechanism
SENS_PTH="/home/devashish_tripathi/projects/sanity_data/sensitivity.pt"

if [ "$USE_NO_NOISE" = true ]; then
    EXTRA_ARGS+=(--no_noise)
fi

if [ "$USE_NO_LOCS" = true ]; then
    EXTRA_ARGS+=(--no_locs)
fi

if [ "$USE_SENSITIVITY" = true ]; then
    EXTRA_ARGS+=(--use_sensitivity --sens_pth "$SENS_PTH")
fi

if [ "$USE_GAUSSIAN" = true ]; then
    EXTRA_ARGS+=(--noise_type G --delta $DELTA)
fi

python -m torch.distributed.run --nproc_per_node=1 train.py \
--yaml_name $YAML_NAME \
--eps_train $EPSILON \
"${EXTRA_ARGS[@]}" \
2>&1 | tee "$LOG_FILE"