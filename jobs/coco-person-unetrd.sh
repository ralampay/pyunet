DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

# Parameters
DEVICE=cuda
GPU_INDEX=0
IMG_WIDTH=256
IMG_HEIGHT=256
INPUT_IMG_DIR=/home/ralampay/workspace/pycocosegmentor/images/coco2017person/images/
MASKED_IMG_DIR=/home/ralampay/workspace/pycocosegmentor/images/coco2017person/masks/
MODEL_FILE=/home/ralampay/workspace/pyunet/models/coco-unet-person-rd.pth
BATCH_SIZE=5
EPOCHS=100
LEARNING_RATE=0.0001
IN_CHANNELS=3
OUT_CHANNELS=2
MODEL_TYPE=unet_rd
CONT=True

python -m pyunet \
  --mode train \
  --device $DEVICE \
  --gpu-index $GPU_INDEX \
  --img-width $IMG_WIDTH \
  --img-height $IMG_HEIGHT \
  --input-img-dir $INPUT_IMG_DIR \
  --input-mask-dir $MASKED_IMG_DIR \
  --model-file $MODEL_FILE \
  --model-type $MODEL_TYPE \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --in-channels $IN_CHANNELS \
  --out-channels $OUT_CHANNELS \
  --cont $CONT

echo "Done"
