# Comprehensive Metrics Documentation for Test-Time Adaptation Evaluation

This document provides a complete mathematical and interpretative guide to all metrics used in our robustness evaluation framework. This documentation is designed to support paper writing and reproducibility.

---

## Table of Contents

1. [Accuracy Metrics](#1-accuracy-metrics)
2. [Prototype-Based Metrics](#2-prototype-based-metrics)
3. [Enhanced Prototype Metrics](#3-enhanced-prototype-metrics)
4. [Computational Efficiency Metrics](#4-computational-efficiency-metrics)
5. [Recommendations for Paper Reporting](#5-recommendations-for-paper-reporting)

---

## 1. Accuracy Metrics

### 1.1 Classification Accuracy

**Definition:**
The standard classification accuracy on corrupted test data.

**Mathematical Formulation:**
\[
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
\]

where:
- \(N\) = total number of test samples
- \(\hat{y}_i\) = predicted class for sample \(i\)
- \(y_i\) = ground truth class for sample \(i\)
- \(\mathbb{1}[\cdot]\) = indicator function

**What it shows:**
- Overall performance on corrupted data
- Direct measure of task success
- **Range:** 0% to 100% (higher is better)

**Category-wise Accuracy:**
Accuracy computed separately for corruption categories:
- **Noise:** gaussian_noise, shot_noise, impulse_noise, speckle_noise
- **Blur:** gaussian_blur, defocus_blur
- **Weather:** fog, frost
- **Digital:** contrast, brightness, elastic_transform, pixelate, jpeg_compression

**Interpretation:**
- Shows which corruption types are most challenging
- Reveals method-specific strengths/weaknesses
- Useful for understanding failure modes

---

## 2. Prototype-Based Metrics

These metrics evaluate how well the prototype-based model maintains its interpretable structure during adaptation.

### 2.1 Prototype Activation Consistency (PAC)

**Definition:**
Measures how similar prototype activations are between clean and adapted (corrupted) images. High PAC indicates that the model preserves its semantic understanding despite corruption.

**Mathematical Formulation:**

For each sample \(i\), compute similarity between clean and adapted activations:

\[
\text{PAC}_i = \text{similarity}(\mathbf{a}_i^{\text{clean}}, \mathbf{a}_i^{\text{adapted}})
\]

where \(\mathbf{a}_i \in \mathbb{R}^P\) is the prototype activation vector (P = number of prototypes).

**Similarity Methods:**

1. **Cosine Similarity (default):**
   \[
   \text{PAC}_i = \frac{\mathbf{a}_i^{\text{clean}} \cdot \mathbf{a}_i^{\text{adapted}}}{\|\mathbf{a}_i^{\text{clean}}\|_2 \|\mathbf{a}_i^{\text{adapted}}\|_2}
   \]

2. **Normalized L2 Distance:**
   \[
   \text{PAC}_i = 1 - \frac{\|\mathbf{a}_i^{\text{clean}} - \mathbf{a}_i^{\text{adapted}}\|_2}{\|\mathbf{a}_i^{\text{clean}}\|_2 + \epsilon}
   \]

3. **Pearson Correlation:**
   \[
   \text{PAC}_i = \text{corr}(\mathbf{a}_i^{\text{clean}}, \mathbf{a}_i^{\text{adapted}})
   \]

**Aggregation:**
\[
\text{PAC}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} \text{PAC}_i
\]

**What it shows:**
- **Semantic preservation:** How well the model maintains its internal representation
- **Stability:** Whether adaptation disrupts the learned prototype structure
- **Range:** 0 to 1 (higher is better, typically 0.8-0.95 for good methods)

**Interpretation:**
- **High PAC (>0.9):** Model maintains interpretable structure, adaptation is conservative
- **Low PAC (<0.7):** Adaptation significantly changes prototype activations, may indicate over-adaptation or instability

---

### 2.2 Prototype Class Alignment (PCA)

**Definition:**
Measures whether the top-activated prototypes belong to the correct class. This evaluates if the model activates semantically relevant prototypes.

**Mathematical Formulation:**

For each sample \(i\) with ground truth class \(y_i\):

1. Get top-k activated prototypes:
   \[
   \mathcal{T}_i = \text{top-k}(\mathbf{a}_i^{\text{adapted}})
   \]

2. Check class membership:
   \[
   \text{PCA}_i = \frac{1}{k} \sum_{p \in \mathcal{T}_i} w_p \cdot \mathbb{1}[\text{class}(p) = y_i]
   \]

where:
- \(w_p\) = activation weight (if `weight_by_activation=True`):
  \[
  w_p = \frac{\exp(a_p)}{\sum_{p' \in \mathcal{T}_i} \exp(a_{p'})}
  \]
- Otherwise: \(w_p = 1/k\) (uniform weighting)

**Aggregation:**
\[
\text{PCA}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} \text{PCA}_i
\]

**What it shows:**
- **Semantic correctness:** Whether activated prototypes match the true class
- **Interpretability quality:** Higher PCA means predictions are based on correct-class prototypes
- **Range:** 0 to 1 (higher is better, typically 0.2-0.5 for good methods)

**Interpretation:**
- **High PCA (>0.4):** Model activates class-relevant prototypes, good interpretability
- **Low PCA (<0.2):** Model activates wrong-class prototypes, poor semantic alignment

---

### 2.3 Prototype Activation Sparsity

**Definition:**
Measures how selective the prototype activations are. High sparsity means only a few prototypes activate strongly, which is desirable for interpretability.

**Mathematical Formulation:**

**Gini Coefficient (primary metric):**

For each sample \(i\) with activation vector \(\mathbf{a}_i\):

1. Sort activations: \(\mathbf{a}_i^{\text{sorted}} = \text{sort}(|\mathbf{a}_i|)\)
2. Compute Gini coefficient:
   \[
   G_i = \frac{n + 1 - 2 \sum_{j=1}^{n} (n+1-j) \cdot a_{i,j}^{\text{sorted}}}{n \cdot \sum_{j=1}^{n} a_{i,j}^{\text{sorted}}}
   \]

where \(n = P\) (number of prototypes).

**Alternative: Active Prototype Count**

\[
\text{Active}_i = \sum_{p=1}^{P} \mathbb{1}[|a_{i,p}| > \tau]
\]

where \(\tau = 0.1\) is the activation threshold.

**Aggregation:**
\[
\text{Sparsity}_{\text{gini}} = \frac{1}{N} \sum_{i=1}^{N} G_i
\]
\[
\text{Sparsity}_{\text{active}} = \frac{1}{N} \sum_{i=1}^{N} \text{Active}_i
\]

**What it shows:**
- **Selectivity:** How many prototypes contribute to each decision
- **Interpretability:** Fewer active prototypes = clearer explanation
- **Range:** 
  - Gini: 0 to 1 (higher = more sparse, typically 0.4-0.5)
  - Active count: 0 to P (lower = more sparse, typically 1500-1800 for P=2000)

**Interpretation:**
- **High sparsity (Gini >0.45):** Selective activation, good interpretability
- **Low sparsity (Gini <0.4):** Many prototypes activate, less interpretable

---

## 3. Enhanced Prototype Metrics

These metrics provide deeper insights into adaptation quality beyond basic prototype metrics.

### 3.1 PCA-Weighted by Importance

**Definition:**
Extension of PCA that weights prototypes by both their activation strength AND their importance for the predicted class (from the last layer weights).

**Mathematical Formulation:**

For each sample \(i\) with ground truth class \(y_i\):

1. Get top-k activated prototypes: \(\mathcal{T}_i = \text{top-k}(\mathbf{a}_i)\)
2. Get importance weights: \(\mathbf{w}_i = |W_{y_i, \mathcal{T}_i}|\) where \(W \in \mathbb{R}^{C \times P}\) is the last layer weight matrix
3. Compute combined contribution:
   \[
   c_p = a_{i,p} \cdot w_{y_i,p} \quad \forall p \in \mathcal{T}_i
   \]
4. Compute weighted alignment:
   \[
   \text{PCA-weighted}_i = \frac{\sum_{p \in \mathcal{T}_i} c_p \cdot \mathbb{1}[\text{class}(p) = y_i]}{\sum_{p \in \mathcal{T}_i} c_p}
   \]

**What it shows:**
- **Relevance-weighted correctness:** Not just if prototypes belong to the class, but if they actually contribute to the prediction
- **Prediction quality:** Higher values mean correct-class prototypes are both activated AND important
- **Range:** 0 to 1 (higher is better, typically 0.7-0.9 for good methods)

**Interpretation:**
- **High PCA-weighted (>0.8):** Model uses relevant, important prototypes for correct predictions
- **Low PCA-weighted (<0.6):** Even if prototypes match class, they may not be important for prediction

---

### 3.2 Calibration Score

**Definition:**
Measures how similar the adapted model's predictions are to the clean model's predictions. This evaluates whether adaptation preserves the model's original decision-making.

**Mathematical Formulation:**

**1. Prediction Agreement:**
\[
\text{Agreement} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i^{\text{adapted}} = \hat{y}_i^{\text{clean}}]
\]

**2. Logit Correlation:**
\[
\text{LogitCorr} = \frac{1}{N} \sum_{i=1}^{N} \frac{\mathbf{l}_i^{\text{adapted}} \cdot \mathbf{l}_i^{\text{clean}}}{\|\mathbf{l}_i^{\text{adapted}}\|_2 \|\mathbf{l}_i^{\text{clean}}\|_2}
\]

where \(\mathbf{l}_i \in \mathbb{R}^C\) is the logit vector (C = number of classes).

**3. Confidence Preservation:**
\[
\text{ConfDiff} = \frac{1}{N} \sum_{i=1}^{N} |\max(\text{softmax}(\mathbf{l}_i^{\text{adapted}})) - \max(\text{softmax}(\mathbf{l}_i^{\text{clean}}))|
\]

**What it shows:**
- **Prediction stability:** Whether adaptation changes predictions
- **Confidence preservation:** Whether adaptation maintains prediction confidence
- **Range:** 
  - Agreement: 0 to 1 (higher is better, typically 0.5-0.8)
  - LogitCorr: -1 to 1 (higher is better, typically 0.8-0.95)

**Interpretation:**
- **High agreement (>0.7):** Adaptation preserves most predictions
- **Low agreement (<0.5):** Adaptation significantly changes predictions (may be good if clean model was wrong)
- **High logit correlation (>0.9):** Model maintains similar decision boundaries

---

### 3.3 Ground Truth Class Contribution Change (GT Δ)

**Definition:**
Measures how the contribution of ground-truth class prototypes changes after adaptation. Positive values indicate improved contribution from correct-class prototypes.

**Mathematical Formulation:**

For each sample \(i\) with ground truth class \(y_i\):

1. Get GT class prototype mask: \(\mathbf{m}_i = \mathbb{1}[\text{class}(p) = y_i] \quad \forall p\)
2. Get importance weights: \(\mathbf{w}_i = |W_{y_i, :}|\) (importance of each prototype for class \(y_i\))
3. Compute weighted contribution:
   \[
   \text{Contrib}_i^{\text{clean}} = \sum_{p=1}^{P} a_{i,p}^{\text{clean}} \cdot m_{i,p} \cdot w_{i,p}
   \]
   \[
   \text{Contrib}_i^{\text{adapted}} = \sum_{p=1}^{P} a_{i,p}^{\text{adapted}} \cdot m_{i,p} \cdot w_{i,p}
   \]
4. Compute relative change:
   \[
   \Delta_i = \frac{\text{Contrib}_i^{\text{adapted}} - \text{Contrib}_i^{\text{clean}}}{\text{Contrib}_i^{\text{clean}} + \epsilon}
   \]

**Aggregation:**
\[
\text{GT}\Delta_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} \Delta_i
\]
\[
\text{GT}\Delta_{\text{improvement}} = \frac{1}{N} \sum_{i=1}^{N} (\text{Contrib}_i^{\text{adapted}} - \text{Contrib}_i^{\text{clean}})
\]

**What it shows:**
- **Adaptation direction:** Whether adaptation increases or decreases correct-class prototype contribution
- **Interpretability improvement:** Positive values mean adaptation makes correct prototypes more important
- **Range:** Typically -0.5 to +0.2 (higher is better, negative is common but less negative is better)

**Interpretation:**
- **Positive GT Δ:** Adaptation increases correct-class prototype contribution (good)
- **Slightly negative GT Δ (-0.1 to 0):** Minimal disruption, acceptable
- **Very negative GT Δ (<-0.3):** Adaptation reduces correct-class contribution (concerning)

---

## 4. Computational Efficiency Metrics

These metrics evaluate the computational cost of different TTA methods.

### 4.1 Time Metrics

**Time per Sample:**
\[
t_{\text{sample}} = \frac{T_{\text{total}}}{N} \times 1000 \quad \text{(milliseconds)}
\]

**Throughput:**
\[
\text{Throughput} = \frac{N}{T_{\text{total}}} \quad \text{(samples/second)}
\]

**Time Overhead (vs. baseline):**
\[
\text{Overhead} = t_{\text{sample}}^{\text{method}} - t_{\text{sample}}^{\text{baseline}}
\]

**What it shows:**
- **Inference speed:** How fast the method processes samples
- **Practical feasibility:** Lower time = more practical for real-time applications
- **Range:** Typically 4-10 ms/sample (lower is better)

---

### 4.2 Parameter Metrics

**Adapted Parameters:**
\[
N_{\text{adapted}} = \sum_{p \in \mathcal{P}_{\text{adapted}}} \text{numel}(p)
\]

**Adaptation Ratio:**
\[
\text{AdaptRatio} = \frac{N_{\text{adapted}}}{N_{\text{total}}}
\]

where \(N_{\text{total}} = \sum_{p \in \mathcal{P}_{\text{all}}} \text{numel}(p)\).

**What it shows:**
- **Adaptation footprint:** How many parameters are modified
- **Selectivity:** Lower ratio = more selective adaptation
- **Range:** Typically 0.0001 to 0.001 (0.01% to 0.1% of total parameters)

---

### 4.3 Adaptation Statistics

**Adaptation Rate:**
\[
\text{AdaptRate} = \frac{N_{\text{adapted\_samples}}}{N_{\text{total\_samples}}}
\]

**Average Updates per Sample:**
\[
\text{Updates/Sample} = \frac{N_{\text{total\_updates}}}{N_{\text{total\_samples}}}
\]

**Steps per Sample:**
\[
\text{Steps/Sample} = \frac{N_{\text{optimizer\_steps}}}{N_{\text{total\_samples}}}
\]

**What it shows:**
- **Selectivity:** How many samples trigger adaptation (for filtering methods like EATA, ProtoTTA)
- **Update frequency:** How often the model is updated
- **Range:** 
  - AdaptRate: 0 to 1 (lower = more selective, typically 0.5-1.0)
  - Updates/Sample: Typically 0.5-1.0 (1.0 = every sample adapted)

---

### 4.4 Memory Usage

**Peak Memory:**
\[
M_{\text{peak}} = \max_{t} M(t) \quad \text{(MB)}
\]

where \(M(t)\) is memory usage at time \(t\).

**What it shows:**
- **Resource requirements:** GPU memory needed
- **Scalability:** Lower memory = can process larger batches
- **Range:** Typically 4-11 GB (lower is better)

---

## 5. Recommendations for Paper Reporting

### 5.1 Essential Metrics (Must Report)

**For Main Results Table:**
1. **Accuracy** (overall and category-wise)
   - Most important metric for readers
   - Easy to interpret and compare

2. **PAC (Prototype Activation Consistency)**
   - Shows semantic preservation
   - Unique to prototype-based models
   - Demonstrates interpretability maintenance

3. **PCA (Prototype Class Alignment)**
   - Shows semantic correctness
   - Validates that prototypes are meaningful

4. **Time per Sample** (or Throughput)
   - Critical for practical applications
   - Shows computational cost

**Rationale:** These four metrics provide a complete picture: performance (accuracy), interpretability (PAC, PCA), and efficiency (time).

---

### 5.2 Secondary Metrics (Report in Supplementary)

**For Detailed Analysis:**
1. **PCA-Weighted**
   - More nuanced than PCA
   - Shows prediction quality, not just class membership
   - Good for ablation studies

2. **Calibration Agreement**
   - Shows prediction stability
   - Useful for understanding adaptation behavior

3. **GT Δ (Ground Truth Contribution Change)**
   - Shows adaptation direction
   - Good for understanding what adaptation does

4. **Sparsity (Gini)**
   - Shows interpretability quality
   - Useful for prototype-based model analysis

5. **Adaptation Rate** (for selective methods)
   - Shows selectivity of filtering methods
   - Important for EATA, ProtoTTA

6. **Adapted Parameters & Ratio**
   - Shows adaptation footprint
   - Useful for efficiency discussion

---

### 5.3 Suggested Paper Structure

**Main Results Section:**
```
Table 1: Performance Comparison
- Accuracy (overall + categories)
- PAC_mean
- PCA_mean  
- Time/Sample (ms)
- Throughput (samples/sec)
```

**Interpretability Analysis Section:**
```
Table 2: Prototype-Based Metrics
- PAC_mean, PAC_std
- PCA_mean, PCA_std
- PCA_weighted_mean
- Sparsity (Gini)
- GT Δ (improvement)
```

**Efficiency Analysis Section:**
```
Table 3: Computational Efficiency
- Time/Sample (ms)
- Throughput (samples/sec)
- Adapted Params (count + %)
- Adaptation Rate (%)
- Peak Memory (MB)
```

---

### 5.4 Visualization Recommendations

**Figure 1: Accuracy Comparison**
- Bar chart: Method vs. Accuracy (overall + categories)
- Shows performance clearly

**Figure 2: PAC vs. Accuracy Scatter**
- X-axis: Accuracy
- Y-axis: PAC
- Each point = one corruption type
- Shows trade-off between performance and preservation

**Figure 3: Efficiency Comparison**
- Bar chart: Method vs. Time/Sample
- Include baseline for reference
- Shows practical feasibility

**Figure 4: Prototype Metrics Heatmap**
- Rows: Methods
- Columns: Metrics (PAC, PCA, PCA-weighted, Sparsity)
- Color intensity = metric value
- Shows comprehensive comparison

---

### 5.5 Key Insights to Highlight

1. **Accuracy vs. Interpretability Trade-off:**
   - Some methods improve accuracy but reduce PAC (over-adaptation)
   - ProtoTTA maintains high PAC while improving accuracy

2. **Efficiency vs. Performance:**
   - SAR is slowest but not always best
   - ProtoTTA achieves good balance

3. **Selectivity Matters:**
   - Methods with lower adaptation rates (ProtoTTA, EATA) are more efficient
   - Selective adaptation is key for practical deployment

4. **Category-Specific Performance:**
   - Different methods excel on different corruption types
   - Noise corruptions are most challenging

---

## 6. Metric Calculation Summary

### Standard Metrics (All Methods)
- Accuracy
- PAC_mean, PAC_std
- PCA_mean, PCA_std
- Sparsity (Gini, Active count)

### Enhanced Metrics (When Available)
- PCA_weighted_mean, PCA_weighted_std
- Calibration_agreement, Calibration_logit_corr
- GT_class_contrib_improvement, GT_class_contrib_change_mean

### Efficiency Metrics (All Methods)
- Time_per_sample_ms
- Throughput_samples_per_sec
- Num_adapted_params, Adaptation_ratio
- Peak_memory_mb

### Adaptation Statistics (TTA Methods Only)
- Adaptation_rate
- Avg_updates_per_sample
- Total_optimizer_steps

---

## 7. Implementation Notes

### Baseline Collection
- All prototype metrics require a clean baseline
- Baseline collected on 1000 clean test samples (configurable)
- Same baseline used for all methods (ensures fair comparison)

### Metric Computation
- All metrics computed per-sample, then aggregated
- Standard deviations provided for uncertainty quantification
- Per-sample values available for detailed analysis

### Efficiency Tracking
- Timing includes full inference + adaptation overhead
- Memory tracking uses `torch.cuda.max_memory_allocated()`
- Batch-level timing for variance analysis

---

## 8. References to Code

- **Prototype Metrics:** `prototype_tta_metrics.py`
- **Enhanced Metrics:** `enhanced_prototype_metrics.py`
- **Efficiency Tracking:** `efficiency_metrics.py`
- **Evaluation Script:** `evaluate_robustness.py`
- **Visualization:** `visualize_proto_metrics.py`

---

## Appendix: Quick Reference Table

| Metric | Range | Higher = Better? | Key Insight |
|--------|-------|------------------|-------------|
| Accuracy | 0-100% | Yes | Overall performance |
| PAC | 0-1 | Yes | Semantic preservation |
| PCA | 0-1 | Yes | Class alignment |
| PCA-Weighted | 0-1 | Yes | Relevance-weighted alignment |
| Sparsity (Gini) | 0-1 | Yes | Interpretability quality |
| Calibration Agreement | 0-1 | Yes | Prediction stability |
| GT Δ | -∞ to +∞ | Yes | Adaptation direction |
| Time/Sample | 0-∞ ms | No | Computational cost |
| Adaptation Rate | 0-1 | Context-dependent | Selectivity |
| Adapted Params | 0-∞ | No | Adaptation footprint |

---

**Last Updated:** 2026-01-30  
**Version:** 1.0
