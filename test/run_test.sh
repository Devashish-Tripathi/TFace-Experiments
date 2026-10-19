CKPT_PATH="/home/devashish_tripathi/projects/sanity_logs/ckpt"
DATA_ROOT="/home/devashish_tripathi/projects/sanity_data/data_bdct"
BIN_NAME="val_dummy.bin"
MODEL_NAME="Validity"
DATA_NAME="Dummy"
BATCH_SIZE=256
RANDOM_SEED=1337
EPSILON=0.5
EPOCH=-1

# TOGGLES
USE_NO_NOISE=false # keep false to mimic baseline code. true means no noise is added
USE_NO_LOCS=true # keep false to mimic baseline code. true means no location is used
USE_SENSITIVITY=true # keep false to mimic baseline code. true means sensitivity is used
USE_GAUSSIAN=false # keep false to mimic baseline code. true means use Gaussian mechanism
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