/*
 * matmul.c - Drop-in replacement for matmul_forward/backward
 * Uses AVX512 asm kernels with OpenMP for parallelism (thread pool, no overhead)
 *
 * Kernels:
 *   matmul_abt.asm: C = A @ B^T  (for forward pass, weight is [OC,C])
 *   matmul_ab.asm:  C = A @ B    (for backward dinp, standard matmul)
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>

// ASM kernel: C[M,N] = A[M,K] @ B[N,K]^T (B stored row-major as [N,K])
// Requires M%3==0, N%4==0, K%16==0
extern void matmul_abt(float* C, const float* A, const float* B, int M, int N, int K);
// Same as matmul_abt, but adds bias[j] during the store.
extern void matmul_abt_bias(float* C, const float* A, const float* B, int M, int N, int K, const float* bias);

#define NUM_THREADS 32

// Drop-in for matmul_forward
// inp[B*T, C], weight[OC, C], bias[OC], out[B*T, OC]
void matmul_forward_fast(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    int M = B * T;

    // Use asm on the largest rectangular region it supports.
    int M_main = (M / 3) * 3;
    int N_main = (OC / 4) * 4;
    int can_use_asm = (M_main > 0) && (N_main > 0) && ((C & 15) == 0);

    if (!can_use_asm) {
        // Fully scalar fallback (with OpenMP).
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < OC; j++) {
                float sum = (bias != NULL) ? bias[j] : 0.0f;
                for (int k = 0; k < C; k++) {
                    sum += inp[i * C + k] * weight[j * C + k];
                }
                out[i * OC + j] = sum;
            }
        }
        return;
    }

    // Compute rows per thread, rounded to multiple of 3
    int rows_per = ((M_main / NUM_THREADS) / 3) * 3;
    if (rows_per < 3) rows_per = 3;

    // Parallelize with larger chunks per thread
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (int t = 0; t < NUM_THREADS; t++) {
        int row_start = t * rows_per;
        int row_end = row_start + rows_per;
        if (t == NUM_THREADS - 1) row_end = M_main;  // last thread gets remainder
        if (row_start >= M_main) continue;
        
        int m = row_end - row_start;
        if (m > 0) {
            if (bias != NULL) {
                matmul_abt_bias(out + row_start * OC,
                                inp + row_start * C,
                                weight,
                                m, N_main, C,
                                bias);
            } else {
                matmul_abt(out + row_start * OC,
                           inp + row_start * C,
                           weight,
                           m, N_main, C);
            }
        }
    }

    // Handle remaining cols (OC % 4) for main rows.
    if (N_main < OC) {
        #pragma omp parallel for
        for (int i = 0; i < M_main; i++) {
            for (int j = N_main; j < OC; j++) {
                float sum = (bias != NULL) ? bias[j] : 0.0f;
                for (int k = 0; k < C; k++) {
                    sum += inp[i * C + k] * weight[j * C + k];
                }
                out[i * OC + j] = sum;
            }
        }
    }

    // Handle remaining rows (M % 3), all columns.
    for (int i = M_main; i < M; i++) {
        for (int j = 0; j < OC; j++) {
            float sum = (bias != NULL) ? bias[j] : 0.0f;
            for (int k = 0; k < C; k++) {
                sum += inp[i * C + k] * weight[j * C + k];
            }
            out[i * OC + j] = sum;
        }
    }
}
