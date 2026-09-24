CKPT_PATH="/rds/projects/c/chenhp-dpface/Training_Updates/BUPT_Baseline/ckpt"
DATA_ROOT="/rds/projects/c/chenhp-dpface/Datasets/LFW/lfw_align_112"
BIN_NAME="lfw.bin"
MODEL_NAME="BUPT_Baseline"
DATA_NAME="LFW112"
BATCH_SIZE=256
RANDOM_SEED=1337
EPSILON=0.5
EPOCH=-1

# TOGGLES
USE_NO_NOISE=false # keep false to mimic baseline code. true means no noise is added
USE_NO_LOCS=false # keep false to mimic baseline code. true means no location is used
USE_SENSITIVITY=false # keep false to mimic baseline code. true means sensitivity is used
SENS_PTH="/rds/projects/c/chenhp-dpface/Datasets/BUPT-CBFace-50/sensitivity.pt"

if [ "$USE_NO_NOISE" = true ]; then
    EXTRA_ARGS+=(--no_noise)
fi

if [ "$USE_NO_LOCS" = true ]; then
    EXTRA_ARGS+=(--no_locs)
fi

if [ "$USE_SENSITIVITY" = true ]; then
    EXTRA_ARGS+=(--use_sensitivity --sens_pth "$SENS_PTH")
fi

python -u test_dctdp.py \
--ckpt_path $CKPT_PATH \
--data_root $DATA_ROOT \
--bin_name $BIN_NAME \
--model_name $MODEL_NAME \
--data_name $DATA_NAME \
--batch_size $BATCH_SIZE \
--random_seed $RANDOM_SEED \
--epsilon $EPSILON \
--gpu_ids 0 \
--epoch $EPOCH \
"${EXTRA_ARGS[@]}"