# sugar-crepe
A faithful benchmark for CLIP-like models' vision-language compositionality.

# Installation

We use [open_clip](https://github.com/mlfoundations/open_clip) for loading pretrained models. Install it with:
```
pip install open_clip_torch
```

We use images of [COCO-2017](https://cocodataset.org/#download) validation set. Download and extract it to anywhere you want, for example, `data/coco/images/val2017/`.


# Usage

Evaluate a pretrained model using the following command:
```
python main_eval.py --model RN50 \ 
    --pretrained openai \
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

To evaluate the 17 pretrained CLIP models included in the paper, run:
```
python main_eval.py --all
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

To evaluate the text models included in the paper ([Vera](https://huggingface.co/liujch1998/vera) & Grammar model), run:
```
python text_model_eval.py
    --output ./output \ 
    --data_root ./data/ \
```
