DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

# Parameters
DEVICE=cuda
GPU_INDEX=0
IMG_WIDTH=128
IMG_HEIGHT=128
INPUT_IMG_DIR=~/workspace/pyunet/notebooks/images/kvasir-capsule/test/images
MASKED_IMG_DIR=~/workspace/pyunet/notebooks/images/kvasir-capsule/test/masks
MODEL_FILE=~/workspace/pyunet/notebooks/images/kvasir-capsule/unet.pth
MODEL_TYPE=unet
SAMPLED_INDEX=9

python -m pyunet \
  --mode sample-pair \
  --img-width $IMG_WIDTH \
  --img-height $IMG_HEIGHT \
  --input-img-dir $INPUT_IMG_DIR \
  --input-mask-dir $MASKED_IMG_DIR \
  --model-file $MODEL_FILE \
  --device cuda \
  --model-type $MODEL_TYPE \
  --sampled-index $SAMPLED_INDEX
