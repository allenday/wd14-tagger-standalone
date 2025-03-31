
forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)

## Features

- Local image tagging with various machine learning models
- Generate tags for single images or entire directories
- Run as standalone CLI tool or as Google Cloud Function
- Support for Camie Tagger and various WD14/Safetensors models
- Optional GPU acceleration with CUDA

## Installation

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Local CLI Mode

```
usage: run.py [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD] [--ext EXT] [--overwrite] [--cpu] [--rawtag] [--recursive] [--exclude-tag t1,t2,t3] [--model MODELNAME]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35)
  --ext EXT             Extension to add to caption file in case of dir option (default is .txt)
  --overwrite           Overwrite caption file if it exists
  --cpu                 Use CPU only
  --rawtag              Use the raw output of the model
  --recursive           Enable recursive file search
  --exclude-tag t1,t2,t3
                        Specify tags to exclude (Need comma-separated list)
  --model MODELNAME     modelname to use for prediction (default is wd14-convnextv2.v1)
```

single file

```
python run.py --file image.jpg
```

batch execution

```
python run.py --dir dir/dir
```

## Support Models

```
# Camie Tagger (released 2025)
python run.py --model camie-tagger --file image.jpg

# SmilingWolf large model. (released 2024)
python run.py --model wd-vit-large-tagger-v3 --file image.jpg
python run.py --model wd-eva02-large-tagger-v3 --file image.jpg

# SmilingWolf v3 model. (released 2024)
python run.py --model wd-v1-4-vit-tagger.v3 --file image.jpg
python run.py --model wd-v1-4-convnext-tagger.v3 --file image.jpg
python run.py --model wd-v1-4-swinv2-tagger.v3 --file image.jpg

# SmilingWolf v2 model. (released 2023)
python run.py --model wd-v1-4-moat-tagger.v2 --file image.jpg
python run.py --model wd14-vit.v2 --file image.jpg
python run.py --model wd14-convnext.v2 --file image.jpg

# SmilingWolf v1 model. (released 2022)
python run.py --model wd14-vit.v1 --file image.jpg
python run.py --model wd14-convnext.v1 --file image.jpg
python run.py --model wd14-convnextv2.v1 --file image.jpg
python run.py --model wd14-swinv2-v1 --file image.jpg

# Z3D-E621-Convnext
python run.py --model z3d-e621-convnext-toynya --file image.jpg
python run.py --model z3d-e621-convnext-silveroxides --file image.jpg

# kiriyamaX model.
python run.py --model mld-caformer.dec-5-97527 --file image.jpg
python run.py --model mld-tresnetd.6-30000 --file image.jpg
```

## Using GPU

Requires CUDA 12.2 and cuDNN8.x.

```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

https://onnxruntime.ai/docs/install/</br>
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## Cloud Function Mode

This project can also be deployed as a Google Cloud Function that receives images via Pub/Sub, analyzes them, and stores both the images and their vector representations for search.

For Cloud Function deployment instructions and usage, see the [Cloud Function README](README-CLOUD.md).

Quick examples for using the Cloud Function:

```bash
# Load environment variables
source .env

# Publish an image to the Cloud Function
python publish_to_pubsub.py $EXAMPLE_IMAGE_PATH \
  --project=$GCP_PROJECT_ID \
  --topic=$PUBSUB_TOPIC

# Verify the image was processed correctly
python scripts/check_all.py $EXAMPLE_IMAGE_PATH
```

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)
