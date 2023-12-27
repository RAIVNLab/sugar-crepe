# <img src="https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugar_crepe.png?raw=true" height="50"> SugarCrepe: A benchmark for faithful vision-language compositionality evaluation

**GPT-4V on SugarCrepe is [here](https://github.com/RAIVNLab/sugar-crepe/tree/main/gpt-4v-results)**

This is the official repository of SugarCrepe, a benchmark for faithful vision-language compositionality evaluation introduced in our paper [SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality](https://arxiv.org/abs/2306.14610).

On SugarCrepe, given an image, a model is required to select the positive caption that correctly describes the image, against another hard negative text distractor that differs from the positive text only by small compositional changes.

## :wrench: Installation

We use [open_clip](https://github.com/mlfoundations/open_clip) for loading pretrained models. Install it with:
```
pip install open_clip_torch
```

We use images of [COCO-2017](https://cocodataset.org/#download) validation set. Download and extract it to anywhere you want, for example, `data/coco/images/val2017/`.


## :keyboard: Usage

Evaluate a pretrained model using the following command:
```python
python main_eval.py --model RN50 \ 
    --pretrained openai \
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

To evaluate the 17 pretrained CLIP models included in the paper, run:
```python
python main_eval.py --all
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

To evaluate the text models included in the paper ([Vera](https://huggingface.co/liujch1998/vera) & Grammar model), run:
```python
python text_model_eval.py
    --output ./output \ 
    --data_root ./data/ \
```

❗:**You can also use SugarCrepe in the [clip-benchmark](https://github.com/LAION-AI/CLIP_benchmark#compositionality-evaluation)**

## :open_book: Why SugarCrepe?

### Biases in existing benchmarks
Many existing benchmarks contain artifacts in hard negatives that can be easily exploited to achieve high performances.
These artifacts, that the hard negatives are "not plausible" and "non-fluent", render the benchmarks unreliable for compositionality evaluation: Blind models, a plausibility estimation model (Vera) and a grammar-scoring model, can outperform state-of-the-art CLIP models on nearly all of these benchmarks.
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/existing_eval.png?raw=true)


### SugarCrepe removes the biases
In SugarCrepe, we remove the artifacts by leveraging ChatGPT to generate plausible and fluent hard negatives, followed by human validation and an adversarial refinement mechanism to maximally reduce the identified biases. We show some comparisons between SugarCrepe and other benchmarks:
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugarcrepe_vs_existing.png?raw=true)


## :bulb: What we found

### Re-evaluating NegCLIP
We find that NegCLIP's improvements on existing benchmarks, e.g., ARO and CREPE, are overestimated, where its improvements are much smaller on SugarCrepe.
The overestimation is particularly large when the test hard negative type matches the one used in training, which we attribute to models' unintentionally overfitting to the artifacts.
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/re_eval.png?raw=true)

The models we trained can be found [here](https://drive.google.com/drive/folders/1n2ZNldxBteltuqx__id43sQCyzq0of09?usp=drive_link).

### Benchmarking pretrained CLIP models
On SugarCrepe, we benchmark 17 pretrained CLIP models and present 4 findings:
- The best pretrained CLIP models demonstrate some compositional understanding but still
have overall large rooms for improvements.
- All models struggle at identifying SWAP hard negatives, regardless of their pertaining dataset
and model size.
- Existing models are object-centric, struggling to compose attributes and relations.
- Models’ performance on SugarCrepe correlates with their ImageNet zero-shot accuracy.
  
![](https://github.com/RAIVNLab/sugar-crepe/blob/main/assets/sugarcrepe_eval.png?raw=true)


## :paperclip: Cite
If you find this repository useful, please consider citing:
```bibtex
@inproceedings{hsieh2023sugarcrepe,
  title={SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ma, Zixian and Kembhavi, Aniruddha and Krishna, Ranjay},
  booktitle={Thirty-Seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```
