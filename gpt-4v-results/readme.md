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

| Split     | GPT-4V (positive-first) | GPT-4V (negative-first) |
|---------|----------|----------|
| swap-att  | 607/666=0.9114 | 593/666=0.8904 |
| swap-obj | 211/246=0.8577 | 198/246=0.8049 |
| add-att | 604/692=0.8728 | 666/692=0.9624 |
| add-obj | 1859/2062=0.9016 | 1918/2062=0.9302 |
| replace-att | 734/788=0.9315 | 740/788=0.9391 |
| replace-obj | 1578/1652=0.9552 | 1604/1652=0.9709 |
| replace-rel | 1240/1406=0.8819 | |

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
