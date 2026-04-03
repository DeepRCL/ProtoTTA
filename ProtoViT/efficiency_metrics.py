"""
Computational Efficiency Metrics for TTA Methods

Tracks and compares computational costs of different TTA methods:
- Inference time (total and per-sample)
- Number of adapted parameters
- Number of optimizer steps
- Memory usage (if available)
- Adaptation overhead vs. normal inference
"""

import time
import torch
import numpy as np
from typing import Dict, Optional, List
from contextlib import contextmanager


class EfficiencyTracker:
    """Track computational efficiency metrics for TTA methods."""
    
    def __init__(self, method_name: str, device: str = 'cuda'):
        self.method_name = method_name
        self.device = device
        
        # Timing
        self.total_time = 0.0
        self.num_samples = 0
        self.batch_times = []
        
        # Parameters
        self.num_adapted_params = 0
        self.total_params = 0
        self.adapted_param_names = []
        
        # Adaptation steps
        self.num_adaptation_steps = 0
        self.total_optimizer_steps = 0
        
        # Memory (optional)
        self.peak_memory_mb = 0.0
        self.track_memory = device.startswith('cuda')
        
        # State
        self._timer_start = None
        self._batch_timer_start = None
    
    def count_adapted_parameters(self, model, adapted_params: Optional[List] = None):
        """
        Count how many parameters are being adapted.
        
        Args:
            model: The model being adapted
            adapted_params: List of parameters being optimized (if None, count all params with requires_grad=True)
        """
        # Total parameters in model
        self.total_params = sum(p.numel() for p in model.parameters())
        
        # Count adapted parameters
        if adapted_params is not None:
            self.num_adapted_params = sum(p.numel() for p in adapted_params if p.requires_grad)
            # Get parameter names if possible
            param_set = set(id(p) for p in adapted_params)
            for name, param in model.named_parameters():
                if id(param) in param_set:
                    self.adapted_param_names.append(name)
        else:
            # Count all parameters with requires_grad=True
            self.num_adapted_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.adapted_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    @contextmanager
    def track_inference(self, batch_size: int = 1):
        """Context manager to track inference time for a batch."""
        # Start timing
        if self.track_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # End timing
            if self.track_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            self.total_time += elapsed
            self.num_samples += batch_size
            self.batch_times.append(elapsed)
            
            # Track memory
            if self.track_memory and torch.cuda.is_available():
                end_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                self.peak_memory_mb = max(self.peak_memory_mb, end_mem)
    
    def record_adaptation_step(self, num_steps: int = 1):
        """Record that adaptation steps were performed."""
        self.num_adaptation_steps += num_steps
        self.total_optimizer_steps += num_steps
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all efficiency metrics as a dictionary."""
        metrics = {
            'method_name': self.method_name,
            'total_time_sec': self.total_time,
            'num_samples': self.num_samples,
            'time_per_sample_ms': (self.total_time / max(self.num_samples, 1)) * 1000,
            'avg_batch_time_ms': np.mean(self.batch_times) * 1000 if self.batch_times else 0.0,
            'std_batch_time_ms': np.std(self.batch_times) * 1000 if self.batch_times else 0.0,
            'throughput_samples_per_sec': self.num_samples / max(self.total_time, 1e-6),
            'num_adapted_params': self.num_adapted_params,
            'total_params': self.total_params,
            'adaptation_ratio': self.num_adapted_params / max(self.total_params, 1),
            'num_adaptation_steps': self.num_adaptation_steps,
            'total_optimizer_steps': self.total_optimizer_steps,
            'steps_per_sample': self.total_optimizer_steps / max(self.num_samples, 1),
        }
        
        if self.track_memory:
            metrics['peak_memory_mb'] = self.peak_memory_mb
        
        return metrics
    
    def get_summary(self) -> str:
        """Get a human-readable summary of efficiency metrics."""
        metrics = self.get_metrics()
        
        summary = f"\n{'='*60}\n"
        summary += f"Efficiency Metrics: {self.method_name}\n"
        summary += f"{'='*60}\n"
        summary += f"Total inference time:     {metrics['total_time_sec']:.2f} sec\n"
        summary += f"Samples processed:        {metrics['num_samples']}\n"
        summary += f"Time per sample:          {metrics['time_per_sample_ms']:.2f} ms\n"
        summary += f"Throughput:               {metrics['throughput_samples_per_sec']:.1f} samples/sec\n"
        summary += f"Avg batch time:           {metrics['avg_batch_time_ms']:.2f} ± {metrics['std_batch_time_ms']:.2f} ms\n"
        summary += f"\n"
        summary += f"Total parameters:         {metrics['total_params']:,}\n"
        summary += f"Adapted parameters:       {metrics['num_adapted_params']:,} ({metrics['adaptation_ratio']*100:.2f}%)\n"
        summary += f"Adaptation steps:         {metrics['num_adaptation_steps']}\n"
        
        if self.track_memory:
            summary += f"Peak memory usage:        {metrics['peak_memory_mb']:.1f} MB\n"
        
        summary += f"{'='*60}\n"
        
        return summary
    
    def __str__(self):
        return self.get_summary()


def compare_efficiency_metrics(trackers: Dict[str, EfficiencyTracker], 
                               baseline_method: str = 'Normal') -> Dict[str, Dict]:
    """
    Compare efficiency metrics across multiple TTA methods.
    
    Args:
        trackers: Dictionary mapping method names to EfficiencyTracker objects
        baseline_method: Method to use as baseline for computing overheads
    
    Returns:
        Dictionary with comparison metrics
    """
    if not trackers:
        return {}
    
    # Get baseline metrics
    baseline_metrics = None
    if baseline_method in trackers:
        baseline_metrics = trackers[baseline_method].get_metrics()
    
    # Collect all metrics
    comparison = {}
    for method_name, tracker in trackers.items():
        metrics = tracker.get_metrics()
        
        # Add overhead metrics if baseline exists
        if baseline_metrics and method_name != baseline_method:
            metrics['time_overhead_vs_baseline_ms'] = (
                metrics['time_per_sample_ms'] - baseline_metrics['time_per_sample_ms']
            )
            metrics['time_overhead_ratio'] = (
                metrics['time_per_sample_ms'] / max(baseline_metrics['time_per_sample_ms'], 1e-6)
            )
            metrics['slowdown_factor'] = metrics['time_overhead_ratio']
        else:
            metrics['time_overhead_vs_baseline_ms'] = 0.0
            metrics['time_overhead_ratio'] = 1.0
            metrics['slowdown_factor'] = 1.0
        
        comparison[method_name] = metrics
    
    return comparison


def print_efficiency_comparison(trackers: Dict[str, EfficiencyTracker], 
                                baseline_method: str = 'Normal',
                                adaptation_stats: Dict[str, Dict] = None):
    """
    Print a formatted comparison table of efficiency metrics.
    
    Args:
        trackers: Dict mapping method names to EfficiencyTracker objects
        baseline_method: Method to use as baseline
        adaptation_stats: Optional dict mapping method names to adaptation_stats dicts
                         (from TTA wrappers like Tent, EATA, SAR, ProtoEntropy)
    """
    comparison = compare_efficiency_metrics(trackers, baseline_method)
    
    if not comparison:
        print("No efficiency metrics to compare.")
        return
    
    print("\n" + "="*120)
    print("COMPUTATIONAL EFFICIENCY COMPARISON")
    print("="*120)
    
    # Header with new columns
    print(f"{'Method':<25} {'Time/Sample':<15} {'Overhead':<20} {'Adapted Params':<20} {'Steps':<12} {'Adapt %':<12} {'Updates/Samp':<15}")
    print(f"{'':25} {'(ms)':<15} {'(vs baseline)':<20} {'(count / %)':<20} {'':12} {'':12} {'':15}")
    print("-"*120)
    
    # Sort by method name, but put baseline first
    methods = sorted(comparison.keys())
    if baseline_method in methods:
        methods.remove(baseline_method)
        methods = [baseline_method] + methods
    
    for method in methods:
        metrics = comparison[method]
        
        time_str = f"{metrics['time_per_sample_ms']:.2f}"
        
        if method == baseline_method:
            overhead_str = "-"
        else:
            overhead_pct = (metrics['slowdown_factor'] - 1) * 100
            overhead_str = f"+{metrics['time_overhead_vs_baseline_ms']:.2f} ms ({overhead_pct:+.1f}%)"
        
        adapted_str = f"{metrics['num_adapted_params']:,} ({metrics['adaptation_ratio']*100:.2f}%)"
        steps_str = f"{metrics['num_adaptation_steps']}"
        
        # Get adaptation rate and updates per sample from adaptation_stats (not from tracker)
        adapt_rate_str = "-"
        updates_per_sample_str = "-"
        
        if adaptation_stats and method in adaptation_stats:
            stats = adaptation_stats[method]
            if 'total_samples' in stats and stats['total_samples'] > 0:
                # Adaptation rate (% of samples adapted)
                adapt_rate = (stats['adapted_samples'] / stats['total_samples']) * 100
                adapt_rate_str = f"{adapt_rate:.1f}%"
                
                # Updates per sample (actual gradient steps per sample)
                updates_per_sample = stats['total_updates'] / stats['total_samples']
                updates_per_sample_str = f"{updates_per_sample:.3f}"
        
        print(f"{method:<25} {time_str:<15} {overhead_str:<20} {adapted_str:<20} {steps_str:<12} {adapt_rate_str:<12} {updates_per_sample_str:<15}")
    
    print("="*120)
    
    # Additional summary statistics
    if baseline_method in comparison:
        print(f"\nBaseline method: {baseline_method}")
        print(f"Baseline throughput: {comparison[baseline_method]['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Find most efficient adaptation method (excluding baseline)
    tta_methods = {k: v for k, v in comparison.items() if k != baseline_method}
    if tta_methods:
        most_efficient = min(tta_methods.items(), key=lambda x: x[1]['time_per_sample_ms'])
        print(f"\nMost efficient TTA method: {most_efficient[0]} ({most_efficient[1]['time_per_sample_ms']:.2f} ms/sample)")
        
        # Method with smallest adaptation footprint
        smallest_footprint = min(tta_methods.items(), key=lambda x: x[1]['num_adapted_params'])
        print(f"Smallest adaptation footprint: {smallest_footprint[0]} ({smallest_footprint[1]['num_adapted_params']:,} params)")
        
        # Method with highest adaptation rate (if available)
        if adaptation_stats:
            adaptive_methods = {}
            for method_name, stats in adaptation_stats.items():
                if method_name != baseline_method and 'total_samples' in stats and stats['total_samples'] > 0:
                    adapt_rate = (stats['adapted_samples'] / stats['total_samples']) * 100
                    adaptive_methods[method_name] = adapt_rate
            
            if adaptive_methods:
                most_selective = min(adaptive_methods.items(), key=lambda x: x[1])
                most_adaptive = max(adaptive_methods.items(), key=lambda x: x[1])
                print(f"Most selective (lowest adapt rate): {most_selective[0]} ({most_selective[1]:.1f}%)")
                print(f"Most adaptive (highest adapt rate): {most_adaptive[0]} ({most_adaptive[1]:.1f}%)")
    
    print()
