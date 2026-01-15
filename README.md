# llm.c (CPU Optimized Fork)

This is a fork of [karpathy/llm.c](https://github.com/karpathy/llm.c) focused specifically on CPU performance optimizations for training GPT-2.

While the original repo focuses on CUDA/GPU implementations, this fork pushes the limits of what's possible on CPU by optimizing the reference C implementation `train_gpt2.c`.

## Optimizations

We have significantly improved the performance of the backward pass operations compared to the vanilla implementation.

### Key Changes
1. **Matrix Multiplication Backward (`matmul_backward`) - ~2x Speedup**
   - Split calculations into separate optimal paths for `dinp` and `dweight`/`dbias`.
   - **`dweight` / `dbias`:** Implemented "time-blocking" (BT blocks) to keep input data in the L2 cache, preventing repeated expensive memory fetches.
   - **`dinp`:** Implemented register-blocked matrix multiplication (8x32 blocks) to maximize register reuse and vector instruction throughput.

2. **Attention Backward (`attention_backward`) - ~58x Speedup**
   - **Algorithmic Improvement:** Replaced the naive O(T³) softmax gradient calculation with an O(T²) linear-time version using the properties of the Softmax derivative (mathematically equivalent to the efficient gradient formulation used in Flash Attention).
   - **Loop Fusion:** Merged multiple passes over the sequence length into fewer passes to improve cache locality, minimizing Memory IO.
   - **Parallelization:** Added OpenMP pragma collapse to parallelize over both Batch and Head dimensions.
   - **Vectorization:** Rewritten inner loops to allow compiler auto-vectorization over the head size dimension.

3. **AdamW Optimizer (`gpt2_update`) - ~10% Speedup**
   - **Parallelization & Vectorization:** Added OpenMP threads and SIMD directives to fully saturate memory bandwidth.
   - **Loop Invariant Hoisting:** Pre-calculated scalar bias correction terms outside the parameter loop to reduce arithmetic intensity.

4. **Matrix Multiplication Forward (`matmul_forward`) - ~1.2x Speedup**
   - **Memory Packing:** Implemented dynamic swizzling/packing of input and weight matrices to improve cache locality. Weights are packed into block-major format to allow contiguous SIMD loading.
   - **Cache Blocking:** Process data in blocks (8 time steps x 32 output channels) to keep working sets within L1/L2 cache.
   - **Vectorization:** Fully vectorized inner loops using AVX instructions (implicit via compiler OMP simd).

5. **Profiling**
   - Added a detailed profiling system to track the execution time of every individual layer (forward and backward passes).
   - Reports `tokens/s` throughput in real-time.

## Performance Comparison

Comparing this optimized version against the vanilla reference implementation on a high-end CPU.

**Hardware:** AMD Ryzen 9 9950X (16 cores, 32 threads)
**Settings:** `OMP_NUM_THREADS=16`

### Batch Size = 4 (Default)

| Version | Total Time (40 steps) | Throughput | Speedup |
|---------|-----------------------|------------|---------|
| Vanilla | 23.37 s | 440 tokens/s | 1.0x |
| **Optimized** | **14.87 s** | **780 tokens/s** | **1.77x** |

### Batch Size = 16

| Version | Total Time (40 steps) | Throughput | Speedup |
|---------|-----------------------|------------|---------|
| Vanilla | 94.42 s | 480 tokens/s | 1.0x |
| **Optimized** | **44.0 s** | **1060 tokens/s** | **2.15x** |

*(Note: "Vanilla" refers to the original `train_gpt2.c` implementation from the parent repo)*

## Usage

1. Download the starter pack (weights and data):
   ```bash
   chmod u+x ./dev/download_starter_pack.sh
   ./dev/download_starter_pack.sh
   ```

2. Compile and run:
   ```bash
   make train_gpt2
   OMP_NUM_THREADS=16 ./train_gpt2
   ```

## Profiling Output Example

At the end of training, you will see a detailed breakdown of where time is spent:

```
--- Profiling Report ---
Matmul Forward:            12.5634 s ( 26.9%)
Matmul Backward (dinp):    14.9213 s ( 31.9%)
Matmul Backward (dw/db):   11.6382 s ( 24.9%)
Attention Forward:          0.2749 s (  0.6%)
Attention Backward:         0.3719 s (  0.8%)
Layernorm Forward:          0.2523 s (  0.5%)
Layernorm Backward:         0.7389 s (  1.6%)
Gelu Forward:               0.6738 s (  1.4%)
Gelu Backward:              0.7948 s (  1.7%)
Residual Forward:           0.2945 s (  0.6%)
Residual Backward:          0.2255 s (  0.5%)
Encoder Forward:            0.0140 s (  0.0%)
Encoder Backward:           0.0085 s (  0.0%)
Crossentropy Forward:       0.0021 s (  0.0%)
Crossentropy Backward:      0.4959 s (  1.1%)
Softmax Forward:            0.7388 s (  1.6%)
AdamW Update:               2.7006 s (  5.8%)
Total Measured Time:       46.7093 s
```

## License

MIT
