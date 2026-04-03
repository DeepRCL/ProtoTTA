# ProtoTTA: ProtoLens vs. ProtoViT

This document summarizes the key architectural and algorithmic differences between the **ProtoTTA** implementation in **ProtoLens** (Text/NLP) and the original **ProtoViT** (Vision).

## 🚀 Overview of Differences

While both implementations share the core philosophy of **Prototype-Aware Test-Time Adaptation**, they diverge significantly to accommodate the differences between vision and language processing, as well as their respective prototype architectures.

| Feature | ProtoLens (Text) | ProtoViT (Vision) |
| :--- | :--- | :--- |
| **Domain** | Natural Language (Sequence) | Computer Vision (Spatial) |
| **Prototype Nature** | **Shared** across all classes | **Class-Specific** (Fixed assignment) |
| **Similarity Metric** | **Cosine Similarity** (Range: `[-1, 1]`) | Distance/RBF (Range: `[0, 1]`) |
| **Sub-prototypes** | No (Global per-prototype similarity) | Yes (Patch-level similarities) |
| **Consensus Strategy** | Aggregates **across prototypes** for reliable sample filtering. | Aggregates **across sub-prototypes** (K patches) for a single prototype. |
| **Primary Loss** | **Directed BCE**: Uses FC weight signs to push supporting protos HIGH and opposing protos LOW. | **Binary Entropy**: Minimizes entropy of prototypes assigned to the predicted class. |
| **Importance Weighting** | Magnitude of FC weights for the predicted class. | Discrete class-prototype identity map. |
| **Backbone Adaptation** | Transformer Attention Biases + LayerNorm. | LayerNorm + Attention Biases (± Prototype Vectors). |

---

## 🧠 Core Algorithmic Shifts

### 1. Directed Prototype Adaptation (ProtoLens)
In **ProtoViT**, each prototype is pre-assigned to a specific class. ProtoTTA simply encourages those "target" prototypes to be more decisive (entropy $\to 0$).

In **ProtoLens**, prototypes are **shared**. A single prototype might support Class A (positive weight) but contradict Class B (negative weight). Therefore, ProtoLens implements a **Directed BCE Loss**:
- For the predicted class, we look at the FC layer weights.
- **Positive Weight**: Pushes prototype similarity toward **1**.
- **Negative Weight**: Pushes prototype similarity toward **0** (after $[-1, 1] \to [0, 1]$ normalization).
- This ensures the adaptation aligns with the model's learned decision logic rather than just forcing arbitrary "peakiness."

### 2. Similarity Normalization
- **ProtoViT** similarities are naturally in $[0, 1]$ due to distance-based kernels.
- **ProtoLens** uses cosine similarity in $[-1, 1]$. To compute entropy, ProtoLens applies a **Temperature-Scaled Sigmoid** to spread out the activations and then normalizes to $[0, 1]$.

### 3. Geometric Filtering Logic
- **ProtoViT** filters samples based on their maximum similarity to any prototype (identifying out-of-distribution or noisy images).
- **ProtoLens** repurposes the "consensus" logic. Since it lacks sub-prototypes (patches), it uses `top_k_mean` or `max` **across different prototypes** to determine if a text sample has a strong enough "semantic anchor" to be worth adapting on.

### 4. Importance Weighting
- In **ProtoViT**, all prototypes for Class $C$ are usually treated as equally important for that class.
- In **ProtoLens**, importance is **continuous**. We weight the loss for each prototype by the **absolute magnitude** of its weight in the FC layer for the predicted class. This prioritizes adapting the components that actually drive the model's current decision.

---

## 🛠 Usage Context

- **ProtoViT** is optimized for restoring spatial focus and handling common image corruptions (noise, blur, etc.).
- **ProtoLens** is optimized for handling linguistic domain shifts (e.g., Yelp $\to$ Amazon), focusing on preserving the nuanced relationship between shared semantic concepts and classification outcomes.
