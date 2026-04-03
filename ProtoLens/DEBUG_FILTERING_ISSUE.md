# Debugging: Why No Samples Are Being Adapted

## The Problem

ProtoTTA shows `0/4000` adapted samples, meaning the geometric filter is filtering out **ALL** samples, even with threshold 0.05.

## What Changed

After the normalization fix, the geometric filtering logic is still using raw `consensus_sims` values, but the threshold might not match the actual range.

## Diagnosis Steps

1. **Check the debug output** - The code now prints:
   - Consensus sims range: [min, max]
   - Threshold value
   - Similarities range

2. **Possible causes:**
   - **Similarities are negative**: If all similarities are negative (e.g., [-0.1, -0.05]), then threshold 0.05 filters everything
   - **Range mismatch**: If actual range is [0, 0.03] but threshold is 0.05, nothing passes
   - **Consensus strategy issue**: If using 'max' and all max values are below threshold

## Quick Fix Options

### Option 1: Disable Geometric Filter (for testing)
```python
--geo_filter False
```

### Option 2: Lower the threshold
If similarities are in a small range like [0, 0.1], try:
```python
--geo_threshold 0.01
```

### Option 3: Check actual similarity ranges
Run with debug output to see what the actual ranges are, then adjust threshold accordingly.

## Expected Output

After running, check the adaptation statistics for:
- `filter_stats.consensus_min` and `consensus_max` - shows actual range
- `filter_stats.avg_similarity` - average consensus similarity
- `filter_stats.filtered_out` - how many were filtered

This will tell you if the threshold needs adjustment.
