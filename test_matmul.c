/*
 * test_matmul.c - Comprehensive test for matmul kernels
 * Tests both matmul_ab (A @ B) and matmul_abt (A @ B^T)
 * Tests single-threaded and multi-threaded performance
 * Compares against train_gpt2.c's matmul_forward implementation
 *
 * Compile:
 *   nasm -f elf64 -O3 -o matmul_ab.o matmul_ab.asm
 *   nasm -f elf64 -O3 -o matmul_abt.o matmul_abt.asm
 *   gcc -O3 -march=native -fopenmp -o test_matmul test_matmul.c matmul.c matmul_ab.o matmul_abt.o -lpthread -lm -z noexecstack
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>

// ----------------------------------------------------------------------------
// train_gpt2.c matmul implementations (for comparison)

void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward_gpt2(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // train_gpt2.c's optimized implementation with loop unrolling
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            float result[LOOP_UNROLL];
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

// ASM kernels
extern void matmul_ab(float* C, const float* A, const float* B, int M, int N, int K);
extern void matmul_abt(float* C, const float* A, const float* B, int M, int N, int K);

// From matmul.c
extern void matmul_forward_fast(float* out, const float* inp, const float* weight, 
                                 const float* bias, int B, int T, int C, int OC);

// ----------------------------------------------------------------------------
// Reference implementations

void ref_matmul_ab(float* C, const float* A, const float* B, int M, int N, int K) {
    // C[M,N] = A[M,K] @ B[K,N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void ref_matmul_abt(float* C, const float* A, const float* B, int M, int N, int K) {
    // C[M,N] = A[M,K] @ B[N,K]^T
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
// Parallel wrappers for single-kernel testing

typedef struct {
    float* C;
    const float* A;
    const float* B;
    int row_start, row_end;
    int N, K;
    int is_transposed;  // 0 = A@B, 1 = A@B^T
} Work;

static void* worker_ab(void* arg) {
    Work* w = (Work*)arg;
    int M = w->row_end - w->row_start;
    if (M > 0) {
        matmul_ab(w->C + w->row_start * w->N,
                  w->A + w->row_start * w->K,
                  w->B, M, w->N, w->K);
    }
    return NULL;
}

static void* worker_abt(void* arg) {
    Work* w = (Work*)arg;
    int M = w->row_end - w->row_start;
    if (M > 0) {
        matmul_abt(w->C + w->row_start * w->N,
                   w->A + w->row_start * w->K,
                   w->B, M, w->N, w->K);
    }
    return NULL;
}

void matmul_ab_parallel(float* C, const float* A, const float* B, 
                        int M, int N, int K, int num_threads) {
    pthread_t threads[num_threads];
    Work work[num_threads];
    int rows_per = (M + num_threads - 1) / num_threads;
    // Round to multiple of 6 for matmul_ab
    rows_per = ((rows_per + 5) / 6) * 6;
    
    int row = 0;
    int t;
    for (t = 0; t < num_threads && row < M; t++) {
        work[t].C = C;
        work[t].A = A;
        work[t].B = B;
        work[t].N = N;
        work[t].K = K;
        work[t].row_start = row;
        work[t].row_end = (row + rows_per > M) ? M : row + rows_per;
        row = work[t].row_end;
        pthread_create(&threads[t], NULL, worker_ab, &work[t]);
    }
    for (int i = 0; i < t; i++) pthread_join(threads[i], NULL);
}

void matmul_abt_parallel(float* C, const float* A, const float* B, 
                         int M, int N, int K, int num_threads) {
    pthread_t threads[num_threads];
    Work work[num_threads];
    int rows_per = (M + num_threads - 1) / num_threads;
    // Round to multiple of 3 for matmul_abt
    rows_per = ((rows_per + 2) / 3) * 3;
    
    int row = 0;
    int t;
    for (t = 0; t < num_threads && row < M; t++) {
        work[t].C = C;
        work[t].A = A;
        work[t].B = B;
        work[t].N = N;
        work[t].K = K;
        work[t].row_start = row;
        work[t].row_end = (row + rows_per > M) ? M : row + rows_per;
        row = work[t].row_end;
        pthread_create(&threads[t], NULL, worker_abt, &work[t]);
    }
    for (int i = 0; i < t; i++) pthread_join(threads[i], NULL);
}

// ----------------------------------------------------------------------------
// Utilities

float randf() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int check_result(const float* got, const float* expected, int n, float tol, const char* name) {
    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    if (max_diff > tol) {
        printf("  FAIL %s: max diff = %.6e at idx %d (got %.6f, expected %.6f)\n",
               name, max_diff, max_idx, got[max_idx], expected[max_idx]);
        return 0;
    }
    printf("  PASS %s: max diff = %.6e\n", name, max_diff);
    return 1;
}

void benchmark(const char* name, void (*fn)(void), int iters, int M, int N, int K) {
    // Warmup
    for (int i = 0; i < 3; i++) fn();
    
    double start = get_time();
    for (int i = 0; i < iters; i++) fn();
    double elapsed = get_time() - start;
    
    double ms = elapsed / iters * 1000;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed / iters) / 1e9;
    printf("  %s: %.2f ms, %.1f GFLOP/s\n", name, ms, gflops);
}

// ----------------------------------------------------------------------------
// Global state for benchmarks
static float *g_A, *g_B, *g_C;
static int g_M, g_N, g_K;

void bench_ref_ab() { ref_matmul_ab(g_C, g_A, g_B, g_M, g_N, g_K); }
void bench_asm_ab() { matmul_ab(g_C, g_A, g_B, g_M, g_N, g_K); }
void bench_asm_ab_8t() { matmul_ab_parallel(g_C, g_A, g_B, g_M, g_N, g_K, 8); }
void bench_asm_ab_16t() { matmul_ab_parallel(g_C, g_A, g_B, g_M, g_N, g_K, 16); }
void bench_asm_ab_32t() { matmul_ab_parallel(g_C, g_A, g_B, g_M, g_N, g_K, 32); }

void bench_ref_abt() { ref_matmul_abt(g_C, g_A, g_B, g_M, g_N, g_K); }
void bench_asm_abt() { matmul_abt(g_C, g_A, g_B, g_M, g_N, g_K); }
void bench_asm_abt_8t() { matmul_abt_parallel(g_C, g_A, g_B, g_M, g_N, g_K, 8); }
void bench_asm_abt_16t() { matmul_abt_parallel(g_C, g_A, g_B, g_M, g_N, g_K, 16); }
void bench_asm_abt_32t() { matmul_abt_parallel(g_C, g_A, g_B, g_M, g_N, g_K, 32); }

// train_gpt2.c benchmarks (weight is [OC,C] = [N,K], need g_B transposed)
static float *g_B_T;  // B transposed for train_gpt2 format
void bench_gpt2_naive() { matmul_forward_naive(g_C, g_A, g_B_T, NULL, 1, g_M, g_K, g_N); }
void bench_gpt2() { matmul_forward_gpt2(g_C, g_A, g_B_T, NULL, 1, g_M, g_K, g_N); }

// ----------------------------------------------------------------------------
// Tests

void test_correctness() {
    printf("\n=== CORRECTNESS TESTS ===\n");
    
    // matmul_ab handles arbitrary dims, matmul_abt needs M%3==0, N%4==0
    printf("\n--- matmul_ab tests (handles any dimension) ---\n");
    {
        int tests[][3] = {
            {6, 16, 8},       // Minimal aligned
            {12, 32, 16},     // Small aligned
            {256, 768, 768},  // GPT-2 like
            {7, 17, 9},       // Odd (edge cases)
            {100, 100, 100},  // Square unaligned
        };
        int num_tests = sizeof(tests) / sizeof(tests[0]);
        
        for (int t = 0; t < num_tests; t++) {
            int M = tests[t][0], N = tests[t][1], K = tests[t][2];
            printf("\nTest M=%d, N=%d, K=%d:\n", M, N, K);
            
            float* A = aligned_alloc(32, M * K * sizeof(float));
            float* B = aligned_alloc(32, K * N * sizeof(float));
            float* C_ref = aligned_alloc(32, M * N * sizeof(float));
            float* C_got = aligned_alloc(32, M * N * sizeof(float));
            
            srand(42 + t);
            for (int i = 0; i < M * K; i++) A[i] = randf();
            for (int i = 0; i < K * N; i++) B[i] = randf();
            
            ref_matmul_ab(C_ref, A, B, M, N, K);
            
            memset(C_got, 0, M * N * sizeof(float));
            matmul_ab(C_got, A, B, M, N, K);
            check_result(C_got, C_ref, M * N, 1e-3f, "matmul_ab (1 thread)");
            
            memset(C_got, 0, M * N * sizeof(float));
            matmul_ab_parallel(C_got, A, B, M, N, K, 8);
            check_result(C_got, C_ref, M * N, 1e-3f, "matmul_ab (8 threads)");
            
            free(A); free(B); free(C_ref); free(C_got);
        }
    }
    
    printf("\n--- matmul_abt tests (requires M%%3==0, N%%4==0, K%%8==0) ---\n");
    {
        int tests[][3] = {
            {6, 16, 8},       // Minimal aligned
            {12, 32, 16},     // Small aligned
            {6, 4, 104},      // Small M,N, large K (104 = 13*8)
            {48, 64, 128},    // Medium
            {252, 768, 768},  // GPT-2 like (252 = 84*3)
            {1020, 768, 768}, // Larger (1020 = 340*3)
        };
        int num_tests = sizeof(tests) / sizeof(tests[0]);
        
        for (int t = 0; t < num_tests; t++) {
            int M = tests[t][0], N = tests[t][1], K = tests[t][2];
            printf("\nTest M=%d, N=%d, K=%d:\n", M, N, K);
            
            float* A = aligned_alloc(32, M * K * sizeof(float));
            float* B = aligned_alloc(32, N * K * sizeof(float));
            float* C_ref = aligned_alloc(32, M * N * sizeof(float));
            float* C_got = aligned_alloc(32, M * N * sizeof(float));
            
            srand(42 + t);
            for (int i = 0; i < M * K; i++) A[i] = randf();
            for (int i = 0; i < N * K; i++) B[i] = randf();
            
            ref_matmul_abt(C_ref, A, B, M, N, K);
            
            memset(C_got, 0, M * N * sizeof(float));
            matmul_abt(C_got, A, B, M, N, K);
            check_result(C_got, C_ref, M * N, 1e-3f, "matmul_abt (1 thread)");
            
            memset(C_got, 0, M * N * sizeof(float));
            matmul_abt_parallel(C_got, A, B, M, N, K, 8);
            check_result(C_got, C_ref, M * N, 1e-3f, "matmul_abt (8 threads)");
            
            free(A); free(B); free(C_ref); free(C_got);
        }
    }
    
    // Test matmul_forward_fast (with bias)
    printf("\n--- matmul_forward_fast test (with bias) ---\n");
    {
        int B = 4, T = 63, C = 768, OC = 768;  // 4*63=252, divisible by 3
        int M = B * T;
        float* inp = aligned_alloc(32, M * C * sizeof(float));
        float* weight = aligned_alloc(32, OC * C * sizeof(float));
        float* bias = aligned_alloc(32, OC * sizeof(float));
        float* out_ref = aligned_alloc(32, M * OC * sizeof(float));
        float* out_got = aligned_alloc(32, M * OC * sizeof(float));
        
        srand(123);
        for (int i = 0; i < M * C; i++) inp[i] = randf();
        for (int i = 0; i < OC * C; i++) weight[i] = randf();
        for (int i = 0; i < OC; i++) bias[i] = randf();
        
        // Reference with bias
        ref_matmul_abt(out_ref, inp, weight, M, OC, C);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < OC; j++)
                out_ref[i * OC + j] += bias[j];
        
        matmul_forward_fast(out_got, inp, weight, bias, B, T, C, OC);
        check_result(out_got, out_ref, M * OC, 1e-3f, "matmul_forward_fast");
        
        free(inp); free(weight); free(bias); free(out_ref); free(out_got);
    }
}

void test_performance() {
    printf("\n=== PERFORMANCE BENCHMARKS ===\n");
    
    int M = 768, N = 1024, K = 768;  
    int iters = 20;
    
    g_A = aligned_alloc(32, M * K * sizeof(float));
    g_B = aligned_alloc(32, K * N * sizeof(float));  // Works for both layouts
    g_C = aligned_alloc(32, M * N * sizeof(float));
    g_M = M; g_N = N; g_K = K;
    
    srand(42);
    for (int i = 0; i < M * K; i++) g_A[i] = randf();
    for (int i = 0; i < K * N; i++) g_B[i] = randf();
    
    printf("\nmatmul_ab (C = A @ B), M=%d, N=%d, K=%d:\n", M, N, K);
    // benchmark("Reference (scalar)", bench_ref_ab, 5, M, N, K);
    benchmark("ASM AVX512 (1 thread)", bench_asm_ab, iters, M, N, K);
    benchmark("ASM AVX512 (8 threads)", bench_asm_ab_8t, iters, M, N, K);
    benchmark("ASM AVX512 (16 threads)", bench_asm_ab_16t, iters, M, N, K);
    benchmark("ASM AVX512 (32 threads)", bench_asm_ab_32t, iters, M, N, K);
    
    printf("\nmatmul_abt (C = A @ B^T), M=%d, N=%d, K=%d:\n", M, N, K);
    // benchmark("Reference (scalar)", bench_ref_abt, 5, M, N, K);
    benchmark("ASM AVX2 (1 thread)", bench_asm_abt, iters, M, N, K);
    benchmark("ASM AVX2 (8 threads)", bench_asm_abt_8t, iters, M, N, K);
    benchmark("ASM AVX2 (16 threads)", bench_asm_abt_16t, iters, M, N, K);
    benchmark("ASM AVX2 (32 threads)", bench_asm_abt_32t, iters, M, N, K);
    
    // Benchmark train_gpt2.c implementation
    // It expects weight in [OC,C] = [N,K] layout, which is B transposed
    g_B_T = aligned_alloc(32, N * K * sizeof(float));
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            g_B_T[j * K + i] = g_B[i * N + j];  // transpose
    
    printf("\ntrain_gpt2.c matmul_forward (C = A @ B^T), M=%d, N=%d, K=%d:\n", M, N, K);
    omp_set_num_threads(1);
    benchmark("matmul_forward_gpt2 (1 thread)", bench_gpt2, iters, M, N, K);
    omp_set_num_threads(8);
    benchmark("matmul_forward_gpt2 (8 threads)", bench_gpt2, iters, M, N, K);
    omp_set_num_threads(16);
    benchmark("matmul_forward_gpt2 (16 threads)", bench_gpt2, iters, M, N, K);
    omp_set_num_threads(32);
    benchmark("matmul_forward_gpt2 (32 threads)", bench_gpt2, iters, M, N, K);
    
    free(g_A); free(g_B); free(g_C); free(g_B_T);
}

int main(int argc, char** argv) {
    printf("Matmul Kernel Test Suite\n");
    printf("========================\n");
    
    test_correctness();
    test_performance();
    
    printf("\n=== ALL TESTS COMPLETE ===\n");
    return 0;
}
