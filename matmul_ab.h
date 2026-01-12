#ifndef MATMUL_AB_H
#define MATMUL_AB_H

// ASM kernel: C[M,N] = A[M,K] @ B[K,N]
// Standard matmul, both A and B row-major
// Requirements: M % 6 == 0, N % 16 == 0 for optimal performance
extern void matmul_ab(float* C, const float* A, const float* B, int M, int N, int K);

#endif // MATMUL_AB_H
