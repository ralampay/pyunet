while getopts n:i:m: flag
do
  case "${flag}" in
    n) DATASET_NAME=${OPTARG};;
    i) INPUT_IMG_DIR=${OPTARG};;
    m) INPUT_MASK_DIR=${OPTARG};;
  esac
done

echo "GENERATE DATASET PARAMETERS:"
echo "========================================="
echo "DATASET_NAME: ${DATASET_NAME}"
echo "INPUT_IMG_DIR: ${INPUT_IMG_DIR}"
echo "INPUT_MASK_DIR: ${INPUT_MASK_DIR}"
echo "========================================="

python -m pyunet \
    --mode generate-dataset \
    --dataset-name $DATASET_NAME \
    --input-img-dir $INPUT_IMG_DIR \
    --input-mask-dir $INPUT_MASK_DIR

# Display contents
tree $DATASET_NAME
