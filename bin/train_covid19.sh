DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

# Parameters
DEVICE=cuda
GPU_INDEX=0
IMG_WIDTH=128
IMG_HEIGHT=128
INPUT_IMG_DIR=/home/ralampay/workspace/pyunet/notebooks/images/covid19ctscandlmulti/training/images
MASKED_IMG_DIR=/home/ralampay/workspace/pyunet/notebooks/images/covid19ctscandlmulti/training/masks
MODEL_FILE=/home/ralampay/workspace/pyunet/notebooks/images/covid19ctscandlmulti/unet_atr.pth
BATCH_SIZE=2
EPOCHS=100
LEARNING_RATE=0.0001
IN_CHANNELS=3
OUT_CHANNELS=4
LOSS_TYPE="CE"
MODEL_TYPE=unet_atr

python -m pyunet \
  --mode train \
  --device $DEVICE \
  --gpu-index $GPU_INDEX \
  --img-width $IMG_WIDTH \
  --img-height $IMG_HEIGHT \
  --input-img-dir $INPUT_IMG_DIR \
  --input-mask-dir $MASKED_IMG_DIR \
  --model-file $MODEL_FILE \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --in-channels $IN_CHANNELS \
  --out-channels $OUT_CHANNELS \
  --model-type $MODEL_TYPE \
  --loss-type $LOSS_TYPE
