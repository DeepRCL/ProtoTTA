# ProtoLens for Test-Time Adaptation (TTA)

**Comprehensive Guide for Integrating ProtoLens into ProtoTTA Project**

---

## 1. Understanding ProtoLens Architecture

### Key Architectural Details

#### **Q: Are prototypes class-specific?**
**A: NO** - Prototypes are **NOT class-specific** in ProtoLens. This is different from ProtoViT!

**How it works:**
```python
# From PLens.py line 77
self.fc = nn.Linear(self.num_prototypes, num_classes)
```

- **Shared prototypes**: You have `num_prototypes` (e.g., 20) prototypes shared across all classes
- **Semantic concepts**: Each prototype represents a semantic concept (e.g., "food quality", "service", "ambience")
- **Classification layer**: The final FC layer learns **class-specific weights** for each prototype
- **Contribution weights**: Prototypes can contribute **positively or negatively** to each class

**Example from Paper (Figure 12 - Steam dataset):**
- Prototype 6: "amazing game" → **+0.935** contribution to positive class
- Prototype 8: "worst game" → **-0.806** contribution to positive class
- Both represent "game quality" but with opposite sentiment contributions

#### **Forward Pass:**
1. Input text → BERT embeddings
2. Extract prototype-relevant spans using DPGMM
3. Compute similarity between text spans and prototypes
4. Aggregate similarities: `[batch_size, num_prototypes]`
5. Linear classification: `logits = self.fc(similarities)` → `[batch_size, num_classes]`

---

## 2. Dataset Information

### Binary Classification Setup

**All datasets are binary sentiment classification:**
- **Label 0**: Negative sentiment
- **Label 1**: Positive sentiment

### Available Datasets

| Dataset | Source | Train Size | Test Size | Domain |
|---------|--------|------------|-----------|--------|
| **Yelp** | Restaurant reviews | 476,537 | 30,000 | Restaurant/Service |
| **Amazon** | Product/Service reviews | ~24,000 | ~6,000 | Mixed Products |
| **IMDB** | Movie reviews | Variable | Variable | Entertainment |
| **Hotel** | Hotel reviews | Not yet available | Not yet available | Hospitality |

### Yelp Dataset Details
- **Total reviews**: ~7 million in raw dataset
- **Processing**: Converted 1-2 stars → Negative (0), 4-5 stars → Positive (1)
- **Excluded**: 3-star reviews (neutral)
- **Distribution**: 
  - Train: 272,811 positive / 203,726 negative (~57% / 43%)
  - Test: 17,189 positive / 12,811 negative (~57% / 43%)

---

## 3. Why ProtoLens is Good for TTA

### ✅ Strong Arguments FOR Using ProtoLens in TTA Project

#### **1. Cross-Modal Generalization**
- ProtoViT: Vision domain (ImageNet)
- ProtoLens: NLP domain (Text sentiment)
- **Shows your TTA method generalizes across modalities!**

#### **2. Real-World Domain Shift**
**Recommended Setup: Yelp → Hotel**
- **Source**: Restaurant reviews (Yelp)
- **Target**: Hotel reviews (Hotel)
- **Domain shift characteristics**:
  - Similar task (sentiment analysis)
  - Different context (dining vs lodging)
  - Overlapping but distinct vocabulary
  - Different aspects matter (food/service vs rooms/amenities)

#### **3. Built-in Adaptation Mechanisms**

ProtoLens already has adaptation components you can leverage:

**a) Prototype Alignment (line 299-338 in PLens.py)**
```python
def align(self):
    # Dynamically aligns prototype vectors with representative sentences
    # This is ALREADY a form of test-time adaptation!
    cosine_sim = torch.einsum('ij,ikj->ik', prototypes, prototype_sentence_emb)
    topk_values, topk_indices = torch.topk(cosine_sim, k=3, dim=-1)
    selected_candidates = torch.mean(torch.stack([...]))
    aligned_prototype_vectors = (selected_candidates - self.prototype_vectors).detach() + self.prototype_vectors
    return aligned_prototype_vectors
```

**b) Diversity Loss**
```python
# line 171
self.diversity_loss = self._diversity_term(self.prototype_vectors)
```

**c) Adaptive Mask Module**
- Uses DPGMM to extract text spans
- Learnable parameters that can be adapted

#### **4. Clear TTA Opportunities**

| Component | What to Adapt | Why It Helps |
|-----------|---------------|--------------|
| **Prototype Vectors** | Update embeddings via gradient descent | Match target domain semantics |
| **FC Layer** | Adapt classification weights | Adjust class-prototype relationships |
| **BatchNorm/LayerNorm** | Update statistics | Handle distribution shift |
| **Prototype Alignment** | Modify alignment mechanism | Better representative selection |
| **DPGMM Parameters** | Adapt span extraction | Target-specific text patterns |

---

## 4. Comparison with TTA Baseline Methods

### Your TTA Method vs Baselines on ProtoLens

#### **Test-Time Training (TTT)**
- **Approach**: Train auxiliary task at test time
- **On ProtoLens**: Could use masked language modeling on test reviews
- **Challenge**: Need to design meaningful auxiliary task for text

#### **TENT (Test Entropy Minimization)**
- **Approach**: Minimize prediction entropy
- **On ProtoLens**: Update model to be more confident on test data
- **Works well when**: Target distribution requires refinement

#### **NOTE (Norm + Entropy)**
- **Approach**: Update normalization layers + entropy minimization
- **On ProtoLens**: Adapt LayerNorm in BERT + classification confidence
- **Similar to**: Your ProtoViT experiments

#### **MEMO (Marginal Entropy Minimization)**
- **Approach**: Minimize entropy over augmented views
- **On ProtoLens**: Text augmentations (synonym replacement, back-translation)
- **Challenge**: Text augmentation less straightforward than image

#### **Your ProtoTTA Method**
Based on your ProtoViT work, you likely adapt:
- Prototype representations
- Normalization statistics
- Classification head
- **Novel contribution**: Prototype-specific adaptation strategy

---

## 5. Training & Evaluation Pipeline

### Step 1: Generate Prototype Initialization Files

**Required files** (per dataset):
```
Datasets/{dataset}/all-mpnet-base-v2/
├── {dataset}_cluster_{K}_centers.npy          # [num_prototypes, 768]
└── {dataset}_cluster_{K}_to_sub_sentence.csv  # [num_prototypes, 40]
```

**Generation script:**
```bash
python generate_prototypes.py \
    --dataset Yelp \
    --num_prototypes 20 \
    --window_size 5 \
    --max_samples 50000 \
    --max_sub_sentences 30000
```

**What it does:**
1. Extracts n-grams (window_size=5) from training texts
2. Computes SentenceTransformer embeddings
3. K-means clustering → prototype centers
4. Finds top-40 representative sub-sentences per prototype

### Step 2: Train Source Model (Yelp)

```bash
python experiment.py \
    -d Yelp \
    -pn 20 \
    -e 25 \
    -bs 16 \
    -lr 0.0005 \
    -i 0  # GPU ID
```

**Expected output:**
- Model checkpoint: `log_folder/Yelp/{experiment_name}/model.pth`
- Training log: `log_folder/Yelp/{experiment_name}/log.txt`
- Validation accuracy: ~85-90% (based on similar work)

### Step 3: Evaluate on Target Domain (Hotel) - No Adaptation

```bash
python evaluate.py \
    --model_path log_folder/Yelp/{experiment}/model.pth \
    --test_dataset Hotel \
    --batch_size 16
```

**Expected drop:** ~10-15% accuracy (domain shift)

### Step 4: Apply TTA Methods

#### **Your ProtoTTA Method:**
```python
# Pseudo-code
def adapt_prototypes(model, test_loader, n_steps=10):
    # Enable gradients only for specific components
    for param in model.parameters():
        param.requires_grad = False
    
    # Adapt prototypes
    model.prototype_vectors.requires_grad = True
    
    # Adapt classification head
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Optimize on test batch
    optimizer = torch.optim.Adam([
        {'params': model.prototype_vectors, 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ])
    
    for step in range(n_steps):
        # Your adaptation loss (e.g., entropy, consistency, etc.)
        loss = compute_adaptation_loss(model, test_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return model
```

#### **Baseline Methods:**
- TENT: Update all parameters, minimize entropy
- NOTE: Update LayerNorm only, minimize entropy
- MEMO: Augment test samples, minimize marginal entropy

### Step 5: Report Results

**Metrics to report:**
- Source-only accuracy (no adaptation)
- Target accuracy after TTA (per method)
- Adaptation time per sample
- Prototype interpretability (qualitative)

**Ablation studies:**
- Which components to adapt? (prototypes, FC, both)
- Number of adaptation steps
- Learning rate sensitivity

---

## 6. Expected Results & Contributions

### Performance Expectations

| Method | Yelp (Source) | Hotel (Target, No Adapt) | Hotel (Target, TTA) | Gain |
|--------|---------------|--------------------------|---------------------|------|
| Source-only | ~88% | ~75% | - | - |
| TENT | - | - | ~78% | +3% |
| NOTE | - | - | ~79% | +4% |
| MEMO | - | - | ~80% | +5% |
| **Your ProtoTTA** | - | - | **~82%** | **+7%** |

*(These are estimates based on typical TTA gains)*

### Your Contributions

1. **Cross-modal TTA**: First to show TTA works on vision AND text prototypes
2. **Prototype-aware adaptation**: Novel strategy for adapting prototype representations
3. **Interpretability preservation**: TTA maintains interpretable explanations
4. **Practical domain shift**: Restaurant → Hotel is real-world applicable

---

## 7. Prototype Interpretability Examples

### What Makes ProtoLens Interpretable?

For each prediction, ProtoLens provides:
1. **Top-K activated prototypes** (e.g., K=3)
2. **Extracted text spans** that activated each prototype
3. **Similarity scores** between spans and prototypes
4. **Contribution weights** to final prediction

### Example (Positive Hotel Review)

**Input:**
> "The room was spacious and clean. Staff was very friendly and helpful. Great location near downtown."

**Top-3 Activated Prototypes:**
| Prototype | Representative Sentences | Extracted Span | Similarity | Contribution |
|-----------|-------------------------|----------------|------------|--------------|
| Proto 5 | "room was spacious", "clean comfortable room" | "room was spacious and clean" | 0.82 | +0.71 |
| Proto 12 | "staff very friendly", "helpful staff" | "Staff was very friendly and helpful" | 0.78 | +0.65 |
| Proto 18 | "great location", "perfect location" | "Great location near downtown" | 0.74 | +0.58 |

**Final Prediction:** Positive (sum of contributions > threshold)

---

## 8. Implementation Checklist

### Phase 1: Setup & Baseline
- [x] Download Yelp dataset
- [x] Prepare Yelp train/test splits (binary classification)
- [ ] Generate prototype initialization for Yelp
- [ ] Generate prototype initialization for Hotel (when available)
- [ ] Fix hardcoded paths in ProtoLens code
- [ ] Train baseline model on Yelp
- [ ] Evaluate on Hotel (source-only, no adaptation)

### Phase 2: Implement TTA Methods
- [ ] Implement TENT baseline
- [ ] Implement NOTE baseline
- [ ] Implement MEMO baseline
- [ ] Implement your ProtoTTA method
- [ ] Create unified evaluation script

### Phase 3: Experiments
- [ ] Run all TTA methods on Yelp→Hotel
- [ ] Hyperparameter tuning
- [ ] Ablation studies
- [ ] Statistical significance tests

### Phase 4: Analysis
- [ ] Quantitative results table
- [ ] Qualitative examples (interpretability)
- [ ] Prototype visualization before/after TTA
- [ ] Error analysis

---

## 9. Potential Challenges & Solutions

### Challenge 1: Missing Hotel Dataset
**Solution:** 
- Use Amazon→Yelp as alternative
- Or collect Hotel reviews from Yelp API
- Or use TripAdvisor hotel reviews

### Challenge 2: Text Augmentation for MEMO
**Solution:**
- Synonym replacement (WordNet)
- Back-translation (English→French→English)
- Paraphrasing with T5
- Random word dropout

### Challenge 3: Slow Training (Large Dataset)
**Solution:**
- Use subset of Yelp (100k samples)
- Early stopping based on validation
- Mixed precision training

### Challenge 4: Prototype Alignment During TTA
**Solution:**
- Freeze alignment or adapt it?
- Update representative sentences from target domain?
- This could be a novel contribution!

---

## 10. Code Modifications Needed

### Fix Hardcoded Paths

**Files to modify:**
- `experiment.py` line 54: Remove hardcoded path
- `utils.py` line 210: Remove hardcoded IMDB path
- `PLens.py`: Already uses relative paths ✓

### Add TTA Evaluation Script

Create `evaluate_tta.py`:
```python
def evaluate_with_tta(model, test_loader, tta_method='tent'):
    """
    Evaluate model with test-time adaptation
    
    Args:
        model: Trained ProtoLens model
        test_loader: Target domain data
        tta_method: 'tent', 'note', 'memo', 'prototta'
    """
    # Implementation here
    pass
```

---

## 11. Timeline Estimate

| Phase | Tasks | Time Estimate |
|-------|-------|---------------|
| Setup | Data prep, prototype generation, baseline training | 2-3 days |
| Implementation | TTA methods + evaluation pipeline | 3-4 days |
| Experiments | Run all methods, hyperparameter tuning | 2-3 days |
| Analysis | Results, visualizations, writing | 2-3 days |
| **Total** | | **~2 weeks** |

---

## 12. Questions & Answers

### Q: Why not class-specific prototypes?
**A:** ProtoLens design choice - semantic concepts are more flexible and interpretable than class-specific patterns. A prototype like "excellent service" can contribute positively to positive class regardless of domain.

### Q: Does TTA make sense without class-specific prototypes?
**A:** YES! In fact, **it's even more interesting**:
- Prototypes must adapt **semantic meanings** across domains
- FC layer must adapt **class-prototype relationships**
- More challenging = more impactful if your method works

### Q: How many samples needed for TTA?
**A:** Typically:
- Per-sample adaptation: 1 sample at a time
- Mini-batch adaptation: 16-32 samples
- Accumulated statistics: 100-1000 samples

### Q: Can we use augmentations for consistency loss?
**A:** YES! Text augmentations:
- Synonym replacement
- Back-translation
- Paraphrasing
- Combine with your prototype consistency loss!

---

## Summary

**ProtoLens is an EXCELLENT choice for your TTA project because:**

1. ✅ Different domain (NLP vs Vision) → shows generalization
2. ✅ Real-world domain shift (Yelp→Hotel)
3. ✅ Built-in adaptation mechanisms to leverage
4. ✅ Interpretability is preserved after adaptation
5. ✅ Binary classification keeps experiments manageable
6. ✅ Prototypes are **not** class-specific, making adaptation more interesting

**Next immediate steps:**
1. Generate prototype initialization files (running now)
2. Train baseline model on Yelp
3. Obtain Hotel dataset or use Amazon as alternative
4. Implement TTA evaluation pipeline

**You're on the right track! This will be a strong contribution to your project.**
