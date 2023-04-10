DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

# Parameters
DEVICE=cuda
GPU_INDEX=0
IMG_WIDTH=128
IMG_HEIGHT=128
INPUT_IMG_DIR=~/Projects/Effect_of_RD_in_UNet_Segmentation/benchmarks/ebhi-seg-polyp-128-01/test/images
MASKED_IMG_DIR=~/Projects/Effect_of_RD_in_UNet_Segmentation/benchmarks/ebhi-seg-polyp-128-01/test/masks
MODEL_FILE=~/Projects/Effect_of_RD_in_UNet_Segmentation/benchmarks/ebhi-seg-polyp-128-01/unet_attn-FL.pth
MODEL_TYPE=unet_attn
SAMPLED_INDEX=12

python -m pyunet \
  --mode sample-pair \
  --img-width $IMG_WIDTH \
  --img-height $IMG_HEIGHT \
  --input-img-dir $INPUT_IMG_DIR \
  --input-mask-dir $MASKED_IMG_DIR \
  --model-file $MODEL_FILE \
  --device $DEVICE \
  --model-type $MODEL_TYPE \
  --sampled-index $SAMPLED_INDEX
