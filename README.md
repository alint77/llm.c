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

4. **Matrix Multiplication Forward (`matmul_forward`)**
   - **Loop Ordering (Cache Blocking):** Swapped the loop order to process Output Channels (`OC`) in the outer loop. This keeps a block of weights (~12-48KB) hot in the L2 cache while streaming the large input activations (12MB), significantly reducing memory bandwidth usage by preventing weight thrashing.
   - **Parallelization:** Parallelized over the `OC` dimension to give each thread a dedicated slice of weights to keep in its private cache.
   - **Memory Packing:** Implemented dynamic swizzling/packing of input and weight matrices into block-major format to allow contiguous SIMD loading.

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
| **Optimized** | **11.3 s** | **1050 tokens/s** | **2.3x** |

### Batch Size = 16

| Version | Total Time (40 steps) | Throughput | Speedup |
|---------|-----------------------|------------|---------|
| Vanilla | 94.42 s | 480 tokens/s | 1.0x |
| **Optimized** | **31.88 s** | **1470 tokens/s** | **3x** |

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

At the end of training, you will see a detailed breakdown of where time is spent (BS=16):

```
--- Profiling Report ---
Matmul Forward:             7.5245 s ( 23.6%)
Matmul Backward (dinp):     8.6685 s ( 27.2%)
Matmul Backward (dw/db):    8.0532 s ( 25.3%)
Attention Forward:          0.3352 s (  1.1%)
Attention Backward:         0.3850 s (  1.2%)
Layernorm Forward:          0.2659 s (  0.8%)
Layernorm Backward:         0.7131 s (  2.2%)
Gelu Forward:               0.6809 s (  2.1%)
Gelu Backward:              0.7474 s (  2.3%)
Residual Forward:           0.3171 s (  1.0%)
Residual Backward:          0.2063 s (  0.6%)
Encoder Forward:            0.0142 s (  0.0%)
Encoder Backward:           0.0077 s (  0.0%)
Crossentropy Forward:       0.0021 s (  0.0%)
Crossentropy Backward:      0.4799 s (  1.5%)
Softmax Forward:            0.7883 s (  2.5%)
AdamW Update:               2.6976 s (  8.5%)
Total Measured Time:       31.8869 s
```

Also here's the profiling results of the original llm.c code:

```
--- Profiling Report ---
Matmul Forward:            12.2042 s ( 12.8%)
Matmul Backward (dinp):    33.9888 s ( 35.6%)
Matmul Backward (dw/db):   18.1879 s ( 19.0%)
Attention Forward:          0.2674 s (  0.3%)
Attention Backward:        23.5974 s ( 24.7%)
Layernorm Forward:          0.2720 s (  0.3%)
Layernorm Backward:         0.7534 s (  0.8%)
Gelu Forward:               0.6676 s (  0.7%)
Gelu Backward:              0.7542 s (  0.8%)
Residual Forward:           0.3018 s (  0.3%)
Residual Backward:          0.2195 s (  0.2%)
Encoder Forward:            0.0136 s (  0.0%)
Encoder Backward:           0.0089 s (  0.0%)
Crossentropy Forward:       0.0021 s (  0.0%)
Crossentropy Backward:      0.4885 s (  0.5%)
Softmax Forward:            0.7398 s (  0.8%)
AdamW Update:               3.0271 s (  3.2%)
Total Measured Time:       95.4942 s
```

## License

MIT
