; matmul_abt.asm - AVX512 C-callable 3x4 kernel for C = A @ B^T
; void matmul_abt(float* C, const float* A, const float* B, int M, int N, int K)
;
; C[M,N] = A[M,K] @ B[N,K]^T
; A is row-major [M,K]: A[i,k] at A + i*K + k
; B is row-major [N,K]: B[j,k] at B + j*K + k  (this is the weight layout!)
; C is row-major [M,N]: C[i,j] at C + i*N + j
;
; Requires: M % 3 == 0, N % 4 == 0, K % 16 == 0
;
; System V AMD64 ABI:
;   rdi = C, rsi = A, rdx = B, rcx = M, r8 = N, r9 = K

section .text
global matmul_abt
global matmul_abt_bias

matmul_abt:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Save params
    mov r12, rdi        ; r12 = C
    mov r13, rsi        ; r13 = A
    mov r14, rdx        ; r14 = B
    mov r15, rcx        ; r15 = M
    mov rbx, r8         ; rbx = N
    mov rcx, r9         ; rcx = K

    ; Precompute K*4 (stride in bytes)
    mov r8, rcx
    shl r8, 2           ; r8 = K*4

    mov r9, r8
    imul r9, 3          ; r9 = K*4*3

    xor r10, r10        ; r10 = i = 0

.loop_a_rows:
    xor r11, r11        ; r11 = j = 0

.loop_b_cols:
    ; Setup pointers for this tile
    ; A_ptr = A + i*K*4
    mov rax, r10
    imul rax, rcx
    shl rax, 2
    lea rsi, [r13 + rax]    ; rsi = &A[i][0]

    ; B_ptr = B + j*K*4
    mov rax, r11
    imul rax, rcx
    shl rax, 2
    lea rdx, [r14 + rax]    ; rdx = &B[j][0]

    ; Zero accumulators (3 rows x 4 cols = 12 zmm registers)
    vxorps zmm15, zmm15, zmm15  ; c[i][j]
    vxorps zmm14, zmm14, zmm14  ; c[i][j+1]
    vxorps zmm13, zmm13, zmm13  ; c[i][j+2]
    vxorps zmm12, zmm12, zmm12  ; c[i][j+3]
    vxorps zmm11, zmm11, zmm11  ; c[i+1][j]
    vxorps zmm10, zmm10, zmm10  ; c[i+1][j+1]
    vxorps zmm9, zmm9, zmm9     ; c[i+1][j+2]
    vxorps zmm8, zmm8, zmm8     ; c[i+1][j+3]
    vxorps zmm7, zmm7, zmm7     ; c[i+2][j]
    vxorps zmm6, zmm6, zmm6     ; c[i+2][j+1]
    vxorps zmm5, zmm5, zmm5     ; c[i+2][j+2]
    vxorps zmm4, zmm4, zmm4     ; c[i+2][j+3]

    ; k loop count - process 16 floats at a time
    mov rax, rcx
    shr rax, 4              ; rax = K/16
    test rax, rax
    jz .do_hsum

.loop_dotprod:
    ; Load 3 rows of A (consecutive in k) - 16 floats each
    vmovups zmm1, [rsi]           ; A[i][k:k+16]
    vmovups zmm2, [rsi + r8]      ; A[i+1][k:k+16]
    vmovups zmm3, [rsi + r8*2]    ; A[i+2][k:k+16]

    ; Load B[j][k:k+16] and FMA with all 3 A rows
    vmovups zmm0, [rdx]           ; B[j][k:k+16]
    vfmadd231ps zmm15, zmm0, zmm1
    vfmadd231ps zmm11, zmm0, zmm2
    vfmadd231ps zmm7, zmm0, zmm3

    ; Load B[j+1][k:k+16]
    vmovups zmm0, [rdx + r8]
    vfmadd231ps zmm14, zmm0, zmm1
    vfmadd231ps zmm10, zmm0, zmm2
    vfmadd231ps zmm6, zmm0, zmm3

    ; Load B[j+2][k:k+16]
    vmovups zmm0, [rdx + r8*2]
    vfmadd231ps zmm13, zmm0, zmm1
    vfmadd231ps zmm9, zmm0, zmm2
    vfmadd231ps zmm5, zmm0, zmm3

    ; Load B[j+3][k:k+16]
    vmovups zmm0, [rdx + r9]
    vfmadd231ps zmm12, zmm0, zmm1
    vfmadd231ps zmm8, zmm0, zmm2
    vfmadd231ps zmm4, zmm0, zmm3

    ; Advance k (both A and B advance along k dimension)
    add rsi, 64             ; A: next 16 floats in row
    add rdx, 64             ; B: next 16 floats in row
    dec rax
    jnz .loop_dotprod

.do_hsum:
    ; ========================================================================
    ; Horizontal sum for 512-bit zmm registers
    ; Each zmm has 16 floats that need to be summed to a single scalar
    ; Strategy: fold 512->256->128->64->32 bits
    ; ========================================================================

    ; --- Row i: zmm15 (c[i][j]), zmm14 (c[i][j+1]), zmm13 (c[i][j+2]), zmm12 (c[i][j+3]) ---
    
    ; zmm15 -> scalar in xmm15[0]
    vextractf64x4 ymm0, zmm15, 1    ; high 256 bits
    vextractf64x4 ymm1, zmm15, 0    ; low 256 bits  
    vaddps ymm15, ymm0, ymm1        ; fold to 256
    vextractf128 xmm0, ymm15, 1     ; high 128
    vaddps xmm15, xmm15, xmm0       ; fold to 128
    vhaddps xmm15, xmm15, xmm15     ; [a+b, c+d, a+b, c+d]
    vhaddps xmm15, xmm15, xmm15     ; [sum, sum, sum, sum]

    ; zmm14 -> scalar in xmm14[0]
    vextractf64x4 ymm0, zmm14, 1
    vextractf64x4 ymm1, zmm14, 0
    vaddps ymm14, ymm0, ymm1
    vextractf128 xmm0, ymm14, 1
    vaddps xmm14, xmm14, xmm0
    vhaddps xmm14, xmm14, xmm14
    vhaddps xmm14, xmm14, xmm14

    ; zmm13 -> scalar in xmm13[0]
    vextractf64x4 ymm0, zmm13, 1
    vextractf64x4 ymm1, zmm13, 0
    vaddps ymm13, ymm0, ymm1
    vextractf128 xmm0, ymm13, 1
    vaddps xmm13, xmm13, xmm0
    vhaddps xmm13, xmm13, xmm13
    vhaddps xmm13, xmm13, xmm13

    ; zmm12 -> scalar in xmm12[0]
    vextractf64x4 ymm0, zmm12, 1
    vextractf64x4 ymm1, zmm12, 0
    vaddps ymm12, ymm0, ymm1
    vextractf128 xmm0, ymm12, 1
    vaddps xmm12, xmm12, xmm0
    vhaddps xmm12, xmm12, xmm12
    vhaddps xmm12, xmm12, xmm12

    ; --- Row i+1: zmm11, zmm10, zmm9, zmm8 ---
    
    vextractf64x4 ymm0, zmm11, 1
    vextractf64x4 ymm1, zmm11, 0
    vaddps ymm11, ymm0, ymm1
    vextractf128 xmm0, ymm11, 1
    vaddps xmm11, xmm11, xmm0
    vhaddps xmm11, xmm11, xmm11
    vhaddps xmm11, xmm11, xmm11

    vextractf64x4 ymm0, zmm10, 1
    vextractf64x4 ymm1, zmm10, 0
    vaddps ymm10, ymm0, ymm1
    vextractf128 xmm0, ymm10, 1
    vaddps xmm10, xmm10, xmm0
    vhaddps xmm10, xmm10, xmm10
    vhaddps xmm10, xmm10, xmm10

    vextractf64x4 ymm0, zmm9, 1
    vextractf64x4 ymm1, zmm9, 0
    vaddps ymm9, ymm0, ymm1
    vextractf128 xmm0, ymm9, 1
    vaddps xmm9, xmm9, xmm0
    vhaddps xmm9, xmm9, xmm9
    vhaddps xmm9, xmm9, xmm9

    vextractf64x4 ymm0, zmm8, 1
    vextractf64x4 ymm1, zmm8, 0
    vaddps ymm8, ymm0, ymm1
    vextractf128 xmm0, ymm8, 1
    vaddps xmm8, xmm8, xmm0
    vhaddps xmm8, xmm8, xmm8
    vhaddps xmm8, xmm8, xmm8

    ; --- Row i+2: zmm7, zmm6, zmm5, zmm4 ---
    
    vextractf64x4 ymm0, zmm7, 1
    vextractf64x4 ymm1, zmm7, 0
    vaddps ymm7, ymm0, ymm1
    vextractf128 xmm0, ymm7, 1
    vaddps xmm7, xmm7, xmm0
    vhaddps xmm7, xmm7, xmm7
    vhaddps xmm7, xmm7, xmm7

    vextractf64x4 ymm0, zmm6, 1
    vextractf64x4 ymm1, zmm6, 0
    vaddps ymm6, ymm0, ymm1
    vextractf128 xmm0, ymm6, 1
    vaddps xmm6, xmm6, xmm0
    vhaddps xmm6, xmm6, xmm6
    vhaddps xmm6, xmm6, xmm6

    vextractf64x4 ymm0, zmm5, 1
    vextractf64x4 ymm1, zmm5, 0
    vaddps ymm5, ymm0, ymm1
    vextractf128 xmm0, ymm5, 1
    vaddps xmm5, xmm5, xmm0
    vhaddps xmm5, xmm5, xmm5
    vhaddps xmm5, xmm5, xmm5

    vextractf64x4 ymm0, zmm4, 1
    vextractf64x4 ymm1, zmm4, 0
    vaddps ymm4, ymm0, ymm1
    vextractf128 xmm0, ymm4, 1
    vaddps xmm4, xmm4, xmm0
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4

    ; ========================================================================
    ; Store results to C
    ; xmm15[0]=c[i][j], xmm14[0]=c[i][j+1], xmm13[0]=c[i][j+2], xmm12[0]=c[i][j+3]
    ; xmm11[0]=c[i+1][j], etc.
    ; ========================================================================
    
    ; C_ptr = C + i*N + j
    mov rax, r10
    imul rax, rbx           ; i * N
    add rax, r11            ; + j
    shl rax, 2              ; * 4 bytes
    lea rdi, [r12 + rax]    ; rdi = &C[i][j]

    ; N stride in bytes
    mov rax, rbx
    shl rax, 2              ; rax = N*4

    ; Pack 4 scalars into xmm and store
    ; Row i: combine xmm15, xmm14, xmm13, xmm12 into one vector
    vinsertps xmm15, xmm15, xmm14, 0x10  ; [c00, c01, _, _]
    vinsertps xmm15, xmm15, xmm13, 0x20  ; [c00, c01, c02, _]
    vinsertps xmm15, xmm15, xmm12, 0x30  ; [c00, c01, c02, c03]
    vmovups [rdi], xmm15

    ; Row i+1
    vinsertps xmm11, xmm11, xmm10, 0x10
    vinsertps xmm11, xmm11, xmm9, 0x20
    vinsertps xmm11, xmm11, xmm8, 0x30
    vmovups [rdi + rax], xmm11

    ; Row i+2
    vinsertps xmm7, xmm7, xmm6, 0x10
    vinsertps xmm7, xmm7, xmm5, 0x20
    vinsertps xmm7, xmm7, xmm4, 0x30
    vmovups [rdi + rax*2], xmm7

    ; Next j
    add r11, 4
    cmp r11, rbx
    jl .loop_b_cols

    ; Next i
    add r10, 3
    cmp r10, r15
    jl .loop_a_rows

    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; matmul_abt_bias.asm - same as matmul_abt but adds bias[j] to each output
; void matmul_abt_bias(float* C, const float* A, const float* B, int M, int N, int K, const float* bias)
;
; bias pointer is the 7th arg on the stack at [rbp+16] under SysV ABI.

matmul_abt_bias:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Save params
    mov r12, rdi        ; r12 = C
    mov r13, rsi        ; r13 = A
    mov r14, rdx        ; r14 = B
    mov r15, rcx        ; r15 = M
    mov rbx, r8         ; rbx = N
    mov rcx, r9         ; rcx = K

    ; Precompute K*4 (stride in bytes)
    mov r8, rcx
    shl r8, 2           ; r8 = K*4

    mov r9, r8
    imul r9, 3          ; r9 = K*4*3

    xor r10, r10        ; r10 = i = 0

.loop_a_rows_bias:
    xor r11, r11        ; r11 = j = 0

.loop_b_cols_bias:
    ; Setup pointers for this tile
    ; A_ptr = A + i*K*4
    mov rax, r10
    imul rax, rcx
    shl rax, 2
    lea rsi, [r13 + rax]    ; rsi = &A[i][0]

    ; B_ptr = B + j*K*4
    mov rax, r11
    imul rax, rcx
    shl rax, 2
    lea rdx, [r14 + rax]    ; rdx = &B[j][0]

    ; Zero accumulators (3 rows x 4 cols = 12 zmm registers)
    vxorps zmm15, zmm15, zmm15  ; c[i][j]
    vxorps zmm14, zmm14, zmm14  ; c[i][j+1]
    vxorps zmm13, zmm13, zmm13  ; c[i][j+2]
    vxorps zmm12, zmm12, zmm12  ; c[i][j+3]
    vxorps zmm11, zmm11, zmm11  ; c[i+1][j]
    vxorps zmm10, zmm10, zmm10  ; c[i+1][j+1]
    vxorps zmm9, zmm9, zmm9     ; c[i+1][j+2]
    vxorps zmm8, zmm8, zmm8     ; c[i+1][j+3]
    vxorps zmm7, zmm7, zmm7     ; c[i+2][j]
    vxorps zmm6, zmm6, zmm6     ; c[i+2][j+1]
    vxorps zmm5, zmm5, zmm5     ; c[i+2][j+2]
    vxorps zmm4, zmm4, zmm4     ; c[i+2][j+3]

    ; k loop count - process 16 floats at a time
    mov rax, rcx
    shr rax, 4              ; rax = K/16
    test rax, rax
    jz .do_hsum_bias

.loop_dotprod_bias:
    ; Load 3 rows of A (consecutive in k) - 16 floats each
    vmovups zmm1, [rsi]           ; A[i][k:k+16]
    vmovups zmm2, [rsi + r8]      ; A[i+1][k:k+16]
    vmovups zmm3, [rsi + r8*2]    ; A[i+2][k:k+16]

    ; Load B[j][k:k+16] and FMA with all 3 A rows
    vmovups zmm0, [rdx]           ; B[j][k:k+16]
    vfmadd231ps zmm15, zmm0, zmm1
    vfmadd231ps zmm11, zmm0, zmm2
    vfmadd231ps zmm7, zmm0, zmm3

    ; Load B[j+1][k:k+16]
    vmovups zmm0, [rdx + r8]
    vfmadd231ps zmm14, zmm0, zmm1
    vfmadd231ps zmm10, zmm0, zmm2
    vfmadd231ps zmm6, zmm0, zmm3

    ; Load B[j+2][k:k+16]
    vmovups zmm0, [rdx + r8*2]
    vfmadd231ps zmm13, zmm0, zmm1
    vfmadd231ps zmm9, zmm0, zmm2
    vfmadd231ps zmm5, zmm0, zmm3

    ; Load B[j+3][k:k+16]
    vmovups zmm0, [rdx + r9]
    vfmadd231ps zmm12, zmm0, zmm1
    vfmadd231ps zmm8, zmm0, zmm2
    vfmadd231ps zmm4, zmm0, zmm3

    ; Advance k (both A and B advance along k dimension)
    add rsi, 64             ; A: next 16 floats in row
    add rdx, 64             ; B: next 16 floats in row
    dec rax
    jnz .loop_dotprod_bias

.do_hsum_bias:
    ; Horizontal sum for 512-bit zmm registers

    ; --- Row i ---
    vextractf64x4 ymm0, zmm15, 1
    vextractf64x4 ymm1, zmm15, 0
    vaddps ymm15, ymm0, ymm1
    vextractf128 xmm0, ymm15, 1
    vaddps xmm15, xmm15, xmm0
    vhaddps xmm15, xmm15, xmm15
    vhaddps xmm15, xmm15, xmm15

    vextractf64x4 ymm0, zmm14, 1
    vextractf64x4 ymm1, zmm14, 0
    vaddps ymm14, ymm0, ymm1
    vextractf128 xmm0, ymm14, 1
    vaddps xmm14, xmm14, xmm0
    vhaddps xmm14, xmm14, xmm14
    vhaddps xmm14, xmm14, xmm14

    vextractf64x4 ymm0, zmm13, 1
    vextractf64x4 ymm1, zmm13, 0
    vaddps ymm13, ymm0, ymm1
    vextractf128 xmm0, ymm13, 1
    vaddps xmm13, xmm13, xmm0
    vhaddps xmm13, xmm13, xmm13
    vhaddps xmm13, xmm13, xmm13

    vextractf64x4 ymm0, zmm12, 1
    vextractf64x4 ymm1, zmm12, 0
    vaddps ymm12, ymm0, ymm1
    vextractf128 xmm0, ymm12, 1
    vaddps xmm12, xmm12, xmm0
    vhaddps xmm12, xmm12, xmm12
    vhaddps xmm12, xmm12, xmm12

    ; --- Row i+1 ---
    vextractf64x4 ymm0, zmm11, 1
    vextractf64x4 ymm1, zmm11, 0
    vaddps ymm11, ymm0, ymm1
    vextractf128 xmm0, ymm11, 1
    vaddps xmm11, xmm11, xmm0
    vhaddps xmm11, xmm11, xmm11
    vhaddps xmm11, xmm11, xmm11

    vextractf64x4 ymm0, zmm10, 1
    vextractf64x4 ymm1, zmm10, 0
    vaddps ymm10, ymm0, ymm1
    vextractf128 xmm0, ymm10, 1
    vaddps xmm10, xmm10, xmm0
    vhaddps xmm10, xmm10, xmm10
    vhaddps xmm10, xmm10, xmm10

    vextractf64x4 ymm0, zmm9, 1
    vextractf64x4 ymm1, zmm9, 0
    vaddps ymm9, ymm0, ymm1
    vextractf128 xmm0, ymm9, 1
    vaddps xmm9, xmm9, xmm0
    vhaddps xmm9, xmm9, xmm9
    vhaddps xmm9, xmm9, xmm9

    vextractf64x4 ymm0, zmm8, 1
    vextractf64x4 ymm1, zmm8, 0
    vaddps ymm8, ymm0, ymm1
    vextractf128 xmm0, ymm8, 1
    vaddps xmm8, xmm8, xmm0
    vhaddps xmm8, xmm8, xmm8
    vhaddps xmm8, xmm8, xmm8

    ; --- Row i+2 ---
    vextractf64x4 ymm0, zmm7, 1
    vextractf64x4 ymm1, zmm7, 0
    vaddps ymm7, ymm0, ymm1
    vextractf128 xmm0, ymm7, 1
    vaddps xmm7, xmm7, xmm0
    vhaddps xmm7, xmm7, xmm7
    vhaddps xmm7, xmm7, xmm7

    vextractf64x4 ymm0, zmm6, 1
    vextractf64x4 ymm1, zmm6, 0
    vaddps ymm6, ymm0, ymm1
    vextractf128 xmm0, ymm6, 1
    vaddps xmm6, xmm6, xmm0
    vhaddps xmm6, xmm6, xmm6
    vhaddps xmm6, xmm6, xmm6

    vextractf64x4 ymm0, zmm5, 1
    vextractf64x4 ymm1, zmm5, 0
    vaddps ymm5, ymm0, ymm1
    vextractf128 xmm0, ymm5, 1
    vaddps xmm5, xmm5, xmm0
    vhaddps xmm5, xmm5, xmm5
    vhaddps xmm5, xmm5, xmm5

    vextractf64x4 ymm0, zmm4, 1
    vextractf64x4 ymm1, zmm4, 0
    vaddps ymm4, ymm0, ymm1
    vextractf128 xmm0, ymm4, 1
    vaddps xmm4, xmm4, xmm0
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4

    ; Store results to C, with bias added
    mov rax, r10
    imul rax, rbx           ; i * N
    add rax, r11            ; + j
    shl rax, 2              ; * 4 bytes
    lea rdi, [r12 + rax]    ; rdi = &C[i][j]

    ; N stride in bytes
    mov rax, rbx
    shl rax, 2              ; rax = N*4

    ; Load bias[j..j+3]
    mov rdx, [rbp + 16]
    vmovups xmm0, [rdx + r11*4]

    ; Row i
    vinsertps xmm15, xmm15, xmm14, 0x10
    vinsertps xmm15, xmm15, xmm13, 0x20
    vinsertps xmm15, xmm15, xmm12, 0x30
    vaddps xmm15, xmm15, xmm0
    vmovups [rdi], xmm15

    ; Row i+1
    vinsertps xmm11, xmm11, xmm10, 0x10
    vinsertps xmm11, xmm11, xmm9, 0x20
    vinsertps xmm11, xmm11, xmm8, 0x30
    vaddps xmm11, xmm11, xmm0
    vmovups [rdi + rax], xmm11

    ; Row i+2
    vinsertps xmm7, xmm7, xmm6, 0x10
    vinsertps xmm7, xmm7, xmm5, 0x20
    vinsertps xmm7, xmm7, xmm4, 0x30
    vaddps xmm7, xmm7, xmm0
    vmovups [rdi + rax*2], xmm7

    ; Next j
    add r11, 4
    cmp r11, rbx
    jl .loop_b_cols_bias

    ; Next i
    add r10, 3
    cmp r10, r15
    jl .loop_a_rows_bias

    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
