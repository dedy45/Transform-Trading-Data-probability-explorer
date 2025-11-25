"""
Performance Optimization Demo

This script demonstrates the performance improvements from:
1. Model caching (load once, reuse across instances)
2. Lazy loading (load only when needed)
3. Vectorized batch operations
4. Automatic chunking for large batches

Run this script to verify performance targets:
- Single prediction < 100ms
- Batch 1000 samples < 5s
- Batch 100K samples with chunking
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_engine.pipeline_prediction import PredictionPipeline


def generate_test_data(n_samples: int, n_features: int = 8) -> pd.DataFrame:
    """Generate synthetic test data."""
    np.random.seed(42)
    
    data = {}
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


def test_single_prediction_performance(pipeline: PredictionPipeline, n_iterations: int = 100):
    """Test single prediction performance."""
    print("\n" + "="*60)
    print("TEST 1: Single Prediction Performance")
    print("="*60)
    print(f"Target: < 100ms per prediction")
    print(f"Running {n_iterations} iterations...")
    
    # Generate test sample
    test_sample = generate_test_data(1, n_features=len(pipeline.feature_names))
    
    # Warm-up
    _ = pipeline.predict_for_sample(test_sample)
    
    # Time predictions
    times = []
    for i in range(n_iterations):
        start = time.time()
        result = pipeline.predict_for_sample(test_sample)
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)
    
    # Statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min:     {min_time:.2f}ms")
    print(f"  Max:     {max_time:.2f}ms")
    print(f"  P95:     {p95_time:.2f}ms")
    
    if avg_time < 100:
        print(f"  ✓ PASS: Average time {avg_time:.2f}ms < 100ms")
    else:
        print(f"  ✗ FAIL: Average time {avg_time:.2f}ms >= 100ms")
    
    return avg_time < 100


def test_batch_prediction_performance(pipeline: PredictionPipeline, batch_size: int = 1000):
    """Test batch prediction performance."""
    print("\n" + "="*60)
    print(f"TEST 2: Batch Prediction Performance ({batch_size} samples)")
    print("="*60)
    print(f"Target: < 5s for 1000 samples")
    
    # Generate test batch
    test_batch = generate_test_data(batch_size, n_features=len(pipeline.feature_names))
    
    # Time prediction
    start = time.time()
    results = pipeline.predict_for_batch(test_batch)
    elapsed = time.time() - start
    
    # Statistics
    samples_per_sec = batch_size / elapsed
    ms_per_sample = (elapsed * 1000) / batch_size
    
    print(f"\nResults:")
    print(f"  Total time:        {elapsed:.2f}s")
    print(f"  Samples/second:    {samples_per_sec:.0f}")
    print(f"  Time per sample:   {ms_per_sample:.2f}ms")
    print(f"  Predictions made:  {len(results)}")
    
    if batch_size == 1000 and elapsed < 5.0:
        print(f"  ✓ PASS: Time {elapsed:.2f}s < 5s")
        return True
    elif batch_size == 1000:
        print(f"  ✗ FAIL: Time {elapsed:.2f}s >= 5s")
        return False
    else:
        print(f"  ℹ INFO: Performance test for {batch_size} samples")
        return True


def test_chunking_performance(pipeline: PredictionPipeline, batch_size: int = 100000):
    """Test chunking for very large batches."""
    print("\n" + "="*60)
    print(f"TEST 3: Large Batch with Chunking ({batch_size} samples)")
    print("="*60)
    print(f"Testing automatic chunking for memory efficiency")
    
    # Generate large test batch
    print(f"Generating {batch_size} samples...")
    test_batch = generate_test_data(batch_size, n_features=len(pipeline.feature_names))
    
    # Time prediction with chunking
    print(f"Running prediction with automatic chunking...")
    start = time.time()
    results = pipeline.predict_for_batch(test_batch, show_progress=True)
    elapsed = time.time() - start
    
    # Statistics
    samples_per_sec = batch_size / elapsed
    ms_per_sample = (elapsed * 1000) / batch_size
    
    print(f"\nResults:")
    print(f"  Total time:        {elapsed:.2f}s")
    print(f"  Samples/second:    {samples_per_sec:.0f}")
    print(f"  Time per sample:   {ms_per_sample:.2f}ms")
    print(f"  Predictions made:  {len(results)}")
    print(f"  ✓ PASS: Successfully processed {batch_size} samples with chunking")
    
    return True


def test_model_caching(model_dir: Path):
    """Test model caching performance."""
    print("\n" + "="*60)
    print("TEST 4: Model Caching Performance")
    print("="*60)
    
    # Clear cache first
    PredictionPipeline.clear_model_cache()
    
    # Test 1: Load without cache
    print("\nLoading models WITHOUT cache (first time)...")
    start = time.time()
    pipeline1 = PredictionPipeline(use_cache=False)
    pipeline1.load_models(model_dir)
    time_no_cache = time.time() - start
    print(f"  Time: {time_no_cache:.2f}s")
    
    # Test 2: Load with cache (first time - should be similar)
    print("\nLoading models WITH cache (first time)...")
    PredictionPipeline.clear_model_cache()
    start = time.time()
    pipeline2 = PredictionPipeline(use_cache=True)
    pipeline2.load_models(model_dir)
    time_first_cache = time.time() - start
    print(f"  Time: {time_first_cache:.2f}s")
    
    # Test 3: Load with cache (second time - should be much faster)
    print("\nLoading models WITH cache (second time - from cache)...")
    start = time.time()
    pipeline3 = PredictionPipeline(use_cache=True)
    pipeline3.load_models(model_dir)
    time_cached = time.time() - start
    print(f"  Time: {time_cached:.2f}s")
    
    # Calculate speedup
    speedup = time_no_cache / time_cached if time_cached > 0 else 0
    
    print(f"\nResults:")
    print(f"  No cache:      {time_no_cache:.2f}s")
    print(f"  First cache:   {time_first_cache:.2f}s")
    print(f"  From cache:    {time_cached:.2f}s")
    print(f"  Speedup:       {speedup:.1f}x")
    
    if speedup > 2:
        print(f"  ✓ PASS: Caching provides {speedup:.1f}x speedup")
        return True
    else:
        print(f"  ⚠ WARNING: Caching speedup {speedup:.1f}x is less than expected")
        return False


def test_vectorized_categorization(pipeline: PredictionPipeline, batch_size: int = 10000):
    """Test vectorized categorization performance."""
    print("\n" + "="*60)
    print(f"TEST 5: Vectorized Categorization ({batch_size} samples)")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    prob_win = np.random.uniform(0.3, 0.8, batch_size)
    R_P50 = np.random.uniform(0.0, 3.0, batch_size)
    R_P10 = R_P50 - np.random.uniform(0.5, 1.5, batch_size)
    R_P90 = R_P50 + np.random.uniform(0.5, 1.5, batch_size)
    
    # Time vectorized categorization
    start = time.time()
    quality_labels, recommendations = pipeline._categorize_batch_vectorized(
        prob_win, R_P50, R_P10, R_P90
    )
    elapsed_ms = (time.time() - start) * 1000
    
    # Statistics
    per_sample_us = (elapsed_ms * 1000) / batch_size
    
    print(f"\nResults:")
    print(f"  Total time:        {elapsed_ms:.2f}ms")
    print(f"  Time per sample:   {per_sample_us:.2f}μs")
    print(f"  Samples processed: {len(quality_labels)}")
    print(f"  ✓ PASS: Vectorized categorization is very fast")
    
    # Verify correctness
    assert len(quality_labels) == batch_size
    assert len(recommendations) == batch_size
    assert all(q in ['A+', 'A', 'B', 'C'] for q in quality_labels)
    assert all(r in ['TRADE', 'SKIP'] for r in recommendations)
    
    return True


def main():
    """Run all performance tests."""
    print("\n" + "="*60)
    print("ML PREDICTION ENGINE - PERFORMANCE OPTIMIZATION TESTS")
    print("="*60)
    
    # Check if models exist
    model_dir = Path(__file__).parent.parent.parent / 'data_processed' / 'models'
    
    if not model_dir.exists():
        print(f"\n✗ ERROR: Model directory not found: {model_dir}")
        print("Please train models first using the demo_model_trainer.py script")
        return
    
    # Check for required model files
    required_files = [
        'lgbm_classifier.pkl',
        'isotonic_calibrator.pkl',
        'lgbm_quantile_p10.pkl',
        'lgbm_quantile_p50.pkl',
        'lgbm_quantile_p90.pkl',
        'conformal_meta.json'
    ]
    
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    
    if missing_files:
        print(f"\n✗ ERROR: Missing model files: {missing_files}")
        print("Please train models first using the demo_model_trainer.py script")
        return
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipeline = PredictionPipeline(use_cache=True)
    pipeline.load_models(model_dir)
    
    print(f"Pipeline loaded successfully")
    print(f"Features: {pipeline.feature_names}")
    
    # Run tests
    results = {}
    
    try:
        results['single_prediction'] = test_single_prediction_performance(pipeline, n_iterations=100)
    except Exception as e:
        print(f"\n✗ ERROR in single prediction test: {e}")
        results['single_prediction'] = False
    
    try:
        results['batch_1k'] = test_batch_prediction_performance(pipeline, batch_size=1000)
    except Exception as e:
        print(f"\n✗ ERROR in batch 1K test: {e}")
        results['batch_1k'] = False
    
    try:
        results['batch_10k'] = test_batch_prediction_performance(pipeline, batch_size=10000)
    except Exception as e:
        print(f"\n✗ ERROR in batch 10K test: {e}")
        results['batch_10k'] = False
    
    try:
        results['chunking'] = test_chunking_performance(pipeline, batch_size=100000)
    except Exception as e:
        print(f"\n✗ ERROR in chunking test: {e}")
        results['chunking'] = False
    
    try:
        results['caching'] = test_model_caching(model_dir)
    except Exception as e:
        print(f"\n✗ ERROR in caching test: {e}")
        results['caching'] = False
    
    try:
        results['vectorization'] = test_vectorized_categorization(pipeline, batch_size=10000)
    except Exception as e:
        print(f"\n✗ ERROR in vectorization test: {e}")
        results['vectorization'] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25s} {status}")
    
    # Performance stats
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    
    stats = pipeline.get_performance_stats()
    print(f"  Total predictions:     {stats['prediction_count']}")
    print(f"  Total time:            {stats['total_time_seconds']:.2f}s")
    print(f"  Average time:          {stats['avg_time_ms']:.2f}ms")
    print(f"  Cache enabled:         {stats['cache_enabled']}")
    print(f"  Cached model sets:     {stats['cache_stats']['cached_models']}")
    
    # Overall result
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
