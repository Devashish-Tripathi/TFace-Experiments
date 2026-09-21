CKPT_PATH="/home/devashish_tripathi/projects/sanity_bdct/ckpt"
DATA_ROOT="/home/devashish_tripathi/projects/sanity_data/data_bdct"
BIN_NAME="val_dummy.bin"
MODEL_NAME="sample"
DATA_NAME="dummy"
BATCH_SIZE=2
RANDOM_SEED=1337
EPSILON=4
EPOCH=-1

# TOGGLES
USE_NO_NOISE=false # keep false to mimic baseline code
USE_NO_LOCS=false # keep false to mimic baseline code
USE_SENSITIVITY=false # keep false to mimic baseline code
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

python test_dctdp.py \
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