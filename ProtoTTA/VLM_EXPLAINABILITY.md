# VLM Explainability for ProtoTTA

This guide explains how to adapt the repository's **VLM-based explainability pipeline** to a new model.

The current implementation is centered on a vision setting in:

- [`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py)
- [`ProtoViT/VLM/vlm.py`](../ProtoViT/VLM/vlm.py)

But the pipeline is more general than one backbone. The core idea is to show a VLM:

- the corrupted input,
- the clean counterpart,
- prototype evidence,
- the method prediction,
- optional localization or activation summaries,

and then ask it to rate or explain whether adaptation restored the right semantic reasoning.

## 1. What the VLM Pipeline Needs

For each evaluation sample, prepare:

- the corrupted input,
- the clean input,
- the ground-truth label,
- the model prediction before adaptation,
- the model prediction after adaptation,
- the top activated prototypes,
- prototype evidence images or rendered evidence artifacts,
- a prompt template describing the task.

For a new **vision** model, the cleanest setup is:

- image file,
- corrupted image file,
- top-k prototype crops or prototype gallery images,
- optional heatmaps or activation overlays.

## 2. Core Requirement: Evidence Must Be Human-Readable

The VLM is not reading raw tensors. It needs evidence that can be rendered:

- prototype image patches,
- prototype crops,
- activation maps,
- bounding boxes,
- overlaid similarity maps.

If your model is not image-based, create a rendered artifact that plays the same role.

Examples:

- for text: render highlighted spans, top prototype phrases, and prediction summaries into an image panel,
- for multimodal models: render each supporting evidence block into a canvas the VLM can inspect.

## 3. Suggested Per-Sample Output Schema

For each sample, save a JSON-style record:

```python
{
    "sample_id": ...,
    "clean_path": ...,
    "corrupted_path": ...,
    "ground_truth": ...,
    "unadapted_prediction": ...,
    "adapted_prediction": ...,
    "top_prototypes": [
        {
            "prototype_id": ...,
            "score": ...,
            "image_path": ...,
            "class_name": ...,
        }
    ],
    "activation_summary": ...,
}
```

This is the bridge between your model and the VLM scorer.

## 4. Required Hooks for a New Model

To plug into a VLM explainability pipeline, implement:

### A. Stable inference-only forward path

The precompute phase should not accidentally continue adapting the model.

### B. Prototype extraction

You need the top-k prototypes per sample after:

- unadapted inference,
- ProtoTTA inference.

### C. Prototype evidence resolution

Your code must know how to map a prototype index to a human-readable artifact:

- prototype image,
- prototype crop,
- rendered text span,
- region visualization.

### D. Sample alignment

The clean and corrupted versions must refer to the same semantic example.

## 5. Generic Workflow

### Step 1: build or load a fixed subset

The repository's current VLM workflow uses a fixed subset so comparisons across methods stay aligned.

This is important because:

- VLM evaluation is expensive,
- you want the same examples across methods,
- you want consistent aggregation and correlation analysis.

### Step 2: precompute method outputs

For each method:

- run inference,
- save predictions,
- save top-k prototype evidence,
- save whether the prediction is correct,
- save any localization summaries you want to expose.

### Step 3: assemble evidence panels

For each sample, create a panel or structured prompt containing:

- clean sample,
- corrupted sample,
- method output,
- evidence prototypes.

### Step 4: call the VLM

The VLM should score or explain:

- whether the adapted method focuses on semantically correct evidence,
- whether ProtoTTA improves over the unadapted baseline,
- whether the rationale is coherent with the final prediction.

### Step 5: aggregate scores

Once the VLM responses are saved, aggregate:

- per-method mean score,
- per-corruption score,
- agreement with PAC / PCA / PCA-W,
- qualitative failure modes.

## 6. Vision-Specific Advice

If your model is image-based, keep the evidence panel simple:

1. clean image,
2. corrupted image,
3. top prototypes for unadapted inference,
4. top prototypes for ProtoTTA inference,
5. optional localization maps.

The VLM should be able to answer:

- which prototype set is more semantically aligned,
- whether adaptation restored the right visual concept,
- whether the selected prototypes support the label.

## 7. Text or Non-Vision Advice

The current `vlm_eval.py` is vision-centric, but the idea still transfers.

For text:

- render the input sentence,
- highlight top activated spans,
- show nearest prototype phrases,
- show before/after predictions,
- save them as an image canvas for the VLM.

For any modality, the principle stays the same:

- convert internal reasoning evidence into a compact, human-readable panel.

## 8. Relation to Metrics

The VLM pipeline should not replace PAC or PCA-W. It complements them.

Recommended use:

- use metrics for scale and reproducibility,
- use VLM scoring for semantic verification and narrative analysis,
- report correlations between the two.

## 9. Current Script to Study

The most complete example is:

- [`ProtoViT/vlm_eval.py`](../ProtoViT/vlm_eval.py)

Important components in that script:

- fixed method ordering,
- subset building,
- precompute caches,
- prototype-image resolution,
- artifact saving,
- VLM scoring.

## 10. Recommended Migration Checklist

When porting VLM explainability to a new model, verify:

- you can extract top-k prototypes,
- you can render prototype evidence,
- you can pair corrupted and clean samples,
- you can save method outputs without further adaptation,
- you can build a fixed subset,
- you can compute correlations with PAC / PCA / PCA-W afterward.
