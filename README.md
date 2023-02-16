# PyUNET

Python based tool for UNet

## Installation and Setup

1. Install dependencies

For `pip` users use:

```
pip install -r requirements.txt
```

Install `pytorch` manually (since currently it's not in the `pip` repositories:

```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

2. Activate the environment

For `venv` users:

```
source env/bin/activate
```

## Modules

### Generate Tiff

Generates a set of tiff images from masked values. Need to supply the unique grayscale values first since the program will convert the original masked colored image (presumed to be png) to grayscale then match it with the `--unique_values` flag.

```
python -m pyunet --mode generate-tiff --unique-values 62 113 137 155 176 194 --input-img-dir ./masks --output-img-dir ./output
```

### Monitor from Camera

Runs pyunet from camera feed.

```
python -m  pyunet --mode monitor --img-height 256 --img-width 256 --display-width 800 --display-height 640 --video 0 --model-file ./model.pth --model-type unet_rd
```
