DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

# Parameters
DEVICE=cuda
GPU_INDEX=0
IMG_WIDTH=128
IMG_HEIGHT=128
INPUT_IMG_DIR=~/Projects/Effect_of_RD_in_UNet_Segmentation/benchmarks/ebhi-seg-polyp-128-01/train/images
MASKED_IMG_DIR=~/Projects/Effect_of_RD_in_UNet_Segmentation/benchmarks/ebhi-seg-polyp-128-01/train/masks
MODEL_FILE=~/Projects/Effect_of_RD_in_UNet_Segmentation/benchmarks/ebhi-seg-polyp-128-01/unet_attn_dp-FL.pth
BATCH_SIZE=2
EPOCHS=100
LEARNING_RATE=0.0001
IN_CHANNELS=3
OUT_CHANNELS=2
LOSS_TYPE=FL
MODEL_TYPE=unet_attn_dp

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
