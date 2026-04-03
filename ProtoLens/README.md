# ProtoLens: Interpretable Text Classification with ProtoTTA

**ProtoLens** is a framework for interpretable text classification that uses prototype-based learning to provide sub-sentence level explanations. This repository also features **ProtoTTA**, a specialized Test-Time Adaptation method adapted from the ProtoViT architecture for NLP tasks.

---

## 🚀 Key Features

- **Fine-Grained Interpretability**: Uses DPGMM (Dirichlet Process Gaussian Mixture Models) for flexible text span extraction.
- **Shared Prototypes**: Learns a compact set of semantic prototypes shared across all classes.
- **ProtoTTA (Test-Time Adaptation)**: Automatically adapts the model to distribution shifts (e.g., Yelp → Amazon) at inference time without needing labels.
- **Sub-Sentence Highlighting**: Visualizes exactly which parts of a sentence triggered a classification.

---

## 🧠 ProtoTTA: ProtoLens vs. ProtoViT

Our Test-Time Adaptation strategy is adapted from the original **ProtoViT** (for Vision) but incorporates several key shifts to handle the nuances of text and shared prototype architectures.

### Summary of Differences

| Feature | ProtoLens (Text) | ProtoViT (Vision) |
| :--- | :--- | :--- |
| **Logic** | **Directed BCE**: Uses FC weight signs to push supporting protos HIGH and opposing protos LOW. | **Binary Entropy**: Minimizes entropy of class-specific prototypes. |
| **Prototype Nature** | **Shared** across classes (Nuanced FC weighting). | **Class-Specific** (Fixed assignment). |
| **Similarity** | **Cosine Similarity** (`[-1, 1]`) + Sigmoid Scaling. | Distance-based (`[0, 1]`). |
| **Importance** | Continuous weighting based on **FC weight magnitude**. | Discrete class-prototype identity map. |

For a detailed deep-dive, see [PROTOTTA_COMPARISON.md](./PROTOTTA_COMPARISON.md).

---

## 🛠 Usage

### 1. Training ProtoLens
Train the base model on your source dataset (e.g., Yelp):
```bash
python experiment.py --dataset yelp --epochs 10
```

### 2. Evaluating with ProtoTTA
Evaluate the model on a shifted dataset (e.g., Amazon-C) using Test-Time Adaptation:
```bash
python run_inference_amazon_c.py \
    --mode proto_tta \
    --geo_filter True \
    --geo_threshold 0.3 \
    --sigmoid_temp 5.0
```

**Key Parameters for ProtoTTA:**
- `--geo_threshold`: Geometric filtering threshold (higher is more selective).
- `--sigmoid_temp`: Temperature for scaling cosine similarities (standard: 5.0).
- `--steps`: Number of adaptation steps per batch (default: 1).

---

## 📂 Repository Structure

- `PLens.py`: Core model implementation.
- `proto_tta.py`: **ProtoTTA** implementation logic.
- `DPGMM.py`: Text span extraction logic.
- `experiment.py`: Main training and evaluation script.
- `run_inference_amazon_c.py`: Robustness evaluation script with TTA support.

---

## 📜 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{protolens2025,
  title={ProtoLens: Interpretable Text Classification via Shared Prototypes},
  author={Abootorabi, Mahdi and et al.},
  journal={arXiv preprint},
  year={2025}
}
```
