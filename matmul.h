#ifndef MATMUL_H
#define MATMUL_H

// Drop-in replacement for matmul_forward
// out[B*T, OC] = inp[B*T, C] @ weight[OC, C]^T + bias[OC]
void matmul_forward_fast(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC);

// TODO: matmul_backward_fast

#endif // MATMUL_H
