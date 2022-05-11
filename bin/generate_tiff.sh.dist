DIR=/home/ralampay/workspace/pyunet/

cd $DIR

source env/bin/activate

UNIQUE_VALUES="0 85 170 255"
INPUT_IMG_DIR=/home/ralampay/Pictures/covid19ctscan/masks
OUTPUT_IMG_DIR=/home/ralampay/Pictures/covid19ctscan/training/masks

python -m pyunet \
  --mode generate-tiff \
  --unique-values $UNIQUE_VALUES \
  --input-img-dir $INPUT_IMG_DIR \
  --output-img-dir $OUTPUT_IMG_DIR
