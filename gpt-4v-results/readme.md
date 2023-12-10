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

| Split     | GPT-4V (positive-first) |
|---------|----------|
| swap-att  | 607/666=0.9114 |
| swap-obj | 211/246=0.8577 |
| add-att | 604/692=0.8728 |
| add-obj | 1859/2062=0.9016 |
| replace-att | 734/788=0.9315 |
| replace-obj | 1578/1652=0.9552 |
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
