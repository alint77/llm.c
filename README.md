# llm.c (CPU Optimized Fork)

This is a fork of [karpathy/llm.c](https://github.com/karpathy/llm.c) focused specifically on CPU performance optimizations for training GPT-2.

While the original repo focuses on CUDA/GPU implementations, this fork pushes the limits of what's possible on CPU by optimizing the reference C implementation `train_gpt2.c`.

## Optimizations

We have significantly improved the performance of the backward pass operations compared to the vanilla implementation.

### Key Changes
1. **Matrix Multiplication Backward (`matmul_backward`) - ~2.5x Speedup**
   - Split calculations into separate optimal paths for `dinp` and `dweight`/`dbias`.
   - **`dweight` / `dbias`:** Implemented a **6x32 blocked register accumulation kernel**. Parallelizes over blocks of 6 Output Channels and loops over blocks of 32 Input Channels, utilizing 12 AVX-512 registers to accumulate results with maximal arithmetic intensity. Keeps `inp` stripes resident in L2 cache.
   - **`dinp`:** Optimized memory access pattern by packing `dout` (gradient of output) into local buffers before the transpose-multiply operation. This enables contiguous access for SIMD instructions.

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
| Vanilla | 23.37 s | 460 tokens/s | 1.0x |
| **Optimized** | **12.3 s** | **980 tokens/s** | **2.1x** |

### Batch Size = 16

| Version | Total Time (40 steps) | Throughput | Speedup |
|---------|-----------------------|------------|---------|
| Vanilla | 94.42 s | 480 tokens/s | 1.0x |
| **Optimized** | **35.2.0 s** | **1400 tokens/s** | **2.9x** |

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
Matmul Forward:            10.1596 s ( 28.8%)
Matmul Backward (dinp):     9.4463 s ( 26.8%)
Matmul Backward (dw/db):    8.0421 s ( 22.8%)
Attention Forward:          0.2879 s (  0.8%)
Attention Backward:         0.3867 s (  1.1%)
Layernorm Forward:          0.2710 s (  0.8%)
Layernorm Backward:         0.7223 s (  2.0%)
Gelu Forward:               0.7362 s (  2.1%)
Gelu Backward:              0.7322 s (  2.1%)
Residual Forward:           0.3241 s (  0.9%)
Residual Backward:          0.2107 s (  0.6%)
Encoder Forward:            0.0139 s (  0.0%)
Encoder Backward:           0.0079 s (  0.0%)
Crossentropy Forward:       0.0022 s (  0.0%)
Crossentropy Backward:      0.4890 s (  1.4%)
Softmax Forward:            0.7364 s (  2.1%)
AdamW Update:               2.7018 s (  7.7%)
Total Measured Time:       35.2703 s
```

## License

MIT
