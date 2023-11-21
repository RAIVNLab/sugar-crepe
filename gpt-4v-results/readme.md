# GPT-4V Results

We are evaluating GPT-4V on SugarCrepe, the result is coming soon!

## Prompt we used to evaluate GPT-4V on SugarCrepe

```python
prompt = f"Which caption best describes the image?\n" \
                 f"(1) {caption1}\n (2) {caption2}\n" \
                 f"Output (1) or (2)."
```

We experiment with two settings: (1) always put positive caption as `caption1` and (2) always put negative caption as `caption1`, which we refer to as `positive-first` and `negative-first` respectively.

## Results

| Split     | GPT-4V |
|---------|----------|
| swap-att  | 607/666=0.9114 |
| swap-obj | 211/246=0.8577 |
| add-att |  |
| add-obj |  |
| replace-att |  |
| replace-obj |  |
| replace-rel | |

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
