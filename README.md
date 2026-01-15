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

3. **Profiling**
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
| **Optimized** | **15.21 s** | **760 tokens/s** | **1.54x** |

### Batch Size = 16

| Version | Total Time (40 steps) | Throughput | Speedup |
|---------|-----------------------|------------|---------|
| Vanilla | 94.42 s | 480 tokens/s | 1.0x |
| **Optimized** | **47.64 s** | **1000 tokens/s** | **2.1x** |

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
Matmul Forward:            12.1843 s ( 26.5%)
Matmul Backward (dinp):    14.1394 s ( 30.8%)
Matmul Backward (dw/db):   11.7800 s ( 25.6%)
Attention Forward:          0.2680 s (  0.6%)
Attention Backward:         0.3644 s (  0.8%)
...
Total Measured Time:       45.9417 s
```

## License

MIT
