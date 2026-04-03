# Dogs Robustness Report

- Model: `output_cosine/Dogs/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth`
- Severity: `5`
- Methods: SAR

## Accuracy Comparison

| Method | Gauss | Shot | Impul | Speck | Noise Avg | Defoc | GBlur | Blur Avg | Brit | Fog | Frost | Weather Avg | Contr | Elast | Jpeg | Pixel | Digital Avg | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAR | 36.7 | 40.9 | 41.8 | 48.1 | 41.9 | 31.5 | 30.0 | 30.8 | 59.6 | 34.4 | 30.6 | 41.5 | 31.8 | 50.3 | 59.5 | 63.1 | 51.2 | 42.9 ± 11.5 |

## Efficiency and Interpretability

| Method | PAC | PCA-W | Prediction Stability | Selection Rate | Rel. Speed |
|---|---:|---:|---:|---:|---:|
| SAR | 93.8 $\pm$ 0.2 | 35.3 $\pm$ 11.3 | 58.0% | 0.7% | N/A |
