DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

# Parameters
DEVICE=cuda
GPU_INDEX=0
INPUT_IMG=/home/ralampay/Pictures/covid19ctscan/training/images/scan_slice01.png
MODEL_FILE=/home/ralampay/workspace/pyunet/models/covid19ctscan-128.pth
#INPUT_IMG=/home/ralampay/workspace/pysplitter/test-images/M-33-7-A-d-2-3_225.png
#MODEL_FILE=/home/ralampay/workspace/pyunet/models/landsat-ai-256.pth

python -m pyunet \
  --mode forward \
  --model-file $MODEL_FILE \
  --input-img $INPUT_IMG \
  --device $DEVICE \
  --gpu-index $GPU_INDEX 
