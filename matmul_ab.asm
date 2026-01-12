; matmul_ab.asm - AVX512 matrix multiplication C = A @ B (with edge case handling)
; void matmul_ab(float* C, const float* A, const float* B, int M, int N, int K)
; C[M,N] = A[M,K] @ B[K,N]
;
; System V AMD64 ABI:
;   rdi = C pointer
;   rsi = A pointer
;   rdx = B pointer
;   rcx = M (rows of A, rows of C)
;   r8  = N (cols of B, cols of C)  
;   r9  = K (cols of A, rows of B)
;
; Uses 6x32 kernel for main body (6 rows x 2 zmm registers), handles arbitrary dimensions

section .text
global matmul_ab

matmul_ab:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 56             ; Local variables
    
    ; Save parameters to stack
    mov [rbp-48], rdi       ; C pointer
    mov [rbp-56], rsi       ; A pointer  
    mov [rbp-64], rdx       ; B pointer
    mov [rbp-72], rcx       ; M
    mov [rbp-80], r8        ; N
    mov [rbp-88], r9        ; K
    
    ; Calculate M_main = (M / 6) * 6
    mov rax, rcx
    mov r10, 6
    xor rdx, rdx
    div r10
    imul rax, 6
    mov [rbp-96], rax       ; M_main
    
    ; Calculate N_main = (N / 32) * 32
    mov rax, r8
    mov r10, 32
    xor rdx, rdx
    div r10
    imul rax, 32
    mov [rbp-104], rax      ; N_main
    
    ; Setup common registers
    mov r12, [rbp-72]       ; r12 = M
    mov rbx, [rbp-80]       ; rbx = N
    mov rcx, [rbp-88]       ; rcx = K
    mov r14, [rbp-56]       ; r14 = A
    mov r13, [rbp-64]       ; r13 = B
    mov r15, [rbp-48]       ; r15 = C
    
    ; Calculate strides (in bytes)
    mov r11, rcx
    shl r11, 2              ; r11 = K * 4 (A row stride)
    mov r10, rbx
    shl r10, 2              ; r10 = N * 4 (B/C row stride)
    
    ; Skip main kernel if no full blocks
    mov rax, [rbp-96]
    test rax, rax
    jz .edge_rows
    mov rax, [rbp-104]
    test rax, rax
    jz .edge_cols_main
    
    ; ========================================================================
    ; MAIN 6x32 KERNEL
    ; ========================================================================
    xor r8, r8              ; r8 = i = 0
    
.loop_a_rows:
    xor r9, r9          ; r9 = j = 0 (col index into B/C)
    
.loop_b_cols:
    ; Zero the accumulators for the 6x32 output tile
    vxorps zmm15, zmm15, zmm15  ; c[i+0][j:j+16]
    vxorps zmm14, zmm14, zmm14  ; c[i+0][j+16:j+32]
    vxorps zmm13, zmm13, zmm13  ; c[i+1][j:j+16]
    vxorps zmm12, zmm12, zmm12  ; c[i+1][j+16:j+32]
    vxorps zmm11, zmm11, zmm11  ; c[i+2][j:j+16]
    vxorps zmm10, zmm10, zmm10  ; c[i+2][j+16:j+32]
    vxorps zmm9, zmm9, zmm9     ; c[i+3][j:j+16]
    vxorps zmm8, zmm8, zmm8     ; c[i+3][j+16:j+32]
    vxorps zmm7, zmm7, zmm7     ; c[i+4][j:j+16]
    vxorps zmm6, zmm6, zmm6     ; c[i+4][j+16:j+32]
    vxorps zmm5, zmm5, zmm5     ; c[i+5][j:j+16]
    vxorps zmm4, zmm4, zmm4     ; c[i+5][j+16:j+32]
    
    ; Calculate A base: A + i * K * 4
    mov rax, r8
    imul rax, rcx       ; i * K
    shl rax, 2          ; i * K * 4
    lea rdi, [r14 + rax] ; rdi = &A[i][0]
    
    xor rax, rax        ; rax = k = 0
    
.loop_dotprod:
    ; Calculate B address: B + k * N * 4 + j * 4
    mov rdx, rax        ; k
    imul rdx, rbx       ; k * N
    shl rdx, 2          ; k * N * 4
    add rdx, r13        ; B base
    
    ; Load B[k][j:j+32]
    vmovups zmm0, [rdx + r9*4]       ; B[k][j:j+16]
    vmovups zmm1, [rdx + r9*4 + 64]  ; B[k][j+16:j+32]
    
    ; Process 6 rows of A
    ; Row 0: A[i][k]
    vbroadcastss zmm2, [rdi]
    vfmadd231ps zmm15, zmm2, zmm0
    vfmadd231ps zmm14, zmm2, zmm1
    
    ; Row 1: A[i+1][k]
    vbroadcastss zmm2, [rdi + r11]
    vfmadd231ps zmm13, zmm2, zmm0
    vfmadd231ps zmm12, zmm2, zmm1
    
    ; Row 2: A[i+2][k]
    lea rsi, [rdi + r11*2]
    vbroadcastss zmm2, [rsi]
    vfmadd231ps zmm11, zmm2, zmm0
    vfmadd231ps zmm10, zmm2, zmm1
    
    ; Row 3: A[i+3][k]
    add rsi, r11
    vbroadcastss zmm2, [rsi]
    vfmadd231ps zmm9, zmm2, zmm0
    vfmadd231ps zmm8, zmm2, zmm1
    
    ; Row 4: A[i+4][k]
    lea rsi, [rdi + r11*4]
    vbroadcastss zmm2, [rsi]
    vfmadd231ps zmm7, zmm2, zmm0
    vfmadd231ps zmm6, zmm2, zmm1
    
    ; Row 5: A[i+5][k]
    add rsi, r11
    vbroadcastss zmm2, [rsi]
    vfmadd231ps zmm5, zmm2, zmm0
    vfmadd231ps zmm4, zmm2, zmm1
    
    ; Advance to next k
    add rdi, 4          ; A pointer advances by 1 float
    inc rax             ; k++
    cmp rax, rcx        ; k < K?
    jl .loop_dotprod
    
.store_c:
    ; Calculate C base: C + i * N * 4 + j * 4
    mov rax, r8         ; i
    imul rax, rbx       ; i * N
    add rax, r9         ; i * N + j
    shl rax, 2          ; (i * N + j) * 4
    add rax, r15        ; C base + offset
    
    ; Store 6 rows of results
    vmovups [rax], zmm15
    vmovups [rax + 64], zmm14
    
    add rax, r10        ; next row
    vmovups [rax], zmm13
    vmovups [rax + 64], zmm12
    
    add rax, r10
    vmovups [rax], zmm11
    vmovups [rax + 64], zmm10
    
    add rax, r10
    vmovups [rax], zmm9
    vmovups [rax + 64], zmm8
    
    add rax, r10
    vmovups [rax], zmm7
    vmovups [rax + 64], zmm6
    
    add rax, r10
    vmovups [rax], zmm5
    vmovups [rax + 64], zmm4
    
    ; Next column block
    add r9, 32          ; j += 32
    cmp r9, [rbp-104]   ; j < N_main?
    jl .loop_b_cols
    
    ; Next row block
    add r8, 6           ; i += 6
    cmp r8, [rbp-96]    ; i < M_main?
    jl .loop_a_rows

    ; ========================================================================
    ; EDGE COLUMNS - cols [N_main, N) for rows [0, M_main)
    ; ========================================================================
.edge_cols_main:
    mov r9, [rbp-104]       ; j = N_main
    cmp r9, rbx             ; j >= N?
    jge .edge_rows
    
    xor r8, r8              ; i = 0
    
.edge_cols_row_loop:
    cmp r8, [rbp-96]
    jge .edge_rows
    
    mov r9, [rbp-104]
    
.edge_cols_col_loop:
    cmp r9, rbx
    jge .edge_cols_next_row
    
    ; Check remaining cols
    mov rax, rbx
    sub rax, r9
    cmp rax, 16
    jl .edge_cols_scalar
    
    ; 1x16 kernel
    vxorps zmm15, zmm15, zmm15
    mov rax, r8
    imul rax, rcx
    shl rax, 2
    lea rdi, [r14 + rax]
    xor rax, rax
.edge_cols_dot16:
    mov rdx, rax
    imul rdx, rbx
    shl rdx, 2
    add rdx, r13
    vmovups zmm0, [rdx + r9*4]
    vbroadcastss zmm2, [rdi]
    vfmadd231ps zmm15, zmm2, zmm0
    add rdi, 4
    inc rax
    cmp rax, rcx
    jl .edge_cols_dot16
    mov rax, r8
    imul rax, rbx
    add rax, r9
    shl rax, 2
    add rax, r15
    vmovups [rax], zmm15
    add r9, 16
    jmp .edge_cols_col_loop
    
.edge_cols_scalar:
    vxorps xmm15, xmm15, xmm15
    mov rax, r8
    imul rax, rcx
    shl rax, 2
    lea rdi, [r14 + rax]
    xor rax, rax
.edge_cols_dot1:
    mov rdx, rax
    imul rdx, rbx
    add rdx, r9
    shl rdx, 2
    add rdx, r13
    vmovss xmm0, [rdx]
    vmovss xmm2, [rdi]
    vfmadd231ss xmm15, xmm2, xmm0
    add rdi, 4
    inc rax
    cmp rax, rcx
    jl .edge_cols_dot1
    mov rax, r8
    imul rax, rbx
    add rax, r9
    shl rax, 2
    add rax, r15
    vmovss [rax], xmm15
    inc r9
    jmp .edge_cols_col_loop
    
.edge_cols_next_row:
    inc r8
    jmp .edge_cols_row_loop

    ; ========================================================================
    ; EDGE ROWS - rows [M_main, M) for all cols
    ; ========================================================================
.edge_rows:
    mov r8, [rbp-96]
    cmp r8, r12
    jge .done
    
.edge_rows_loop:
    cmp r8, r12
    jge .done
    xor r9, r9
    
.edge_rows_col_loop:
    cmp r9, rbx
    jge .edge_rows_next
    
    mov rax, rbx
    sub rax, r9
    cmp rax, 32
    jl .edge_rows_check16
    
    ; 1x32 kernel
    vxorps zmm15, zmm15, zmm15
    vxorps zmm14, zmm14, zmm14
    mov rax, r8
    imul rax, rcx
    shl rax, 2
    lea rdi, [r14 + rax]
    xor rax, rax
.edge_rows_dot32:
    mov rdx, rax
    imul rdx, rbx
    shl rdx, 2
    add rdx, r13
    vmovups zmm0, [rdx + r9*4]
    vmovups zmm1, [rdx + r9*4 + 64]
    vbroadcastss zmm2, [rdi]
    vfmadd231ps zmm15, zmm2, zmm0
    vfmadd231ps zmm14, zmm2, zmm1
    add rdi, 4
    inc rax
    cmp rax, rcx
    jl .edge_rows_dot32
    mov rax, r8
    imul rax, rbx
    add rax, r9
    shl rax, 2
    add rax, r15
    vmovups [rax], zmm15
    vmovups [rax + 64], zmm14
    add r9, 32
    jmp .edge_rows_col_loop
    
.edge_rows_check16:
    cmp rax, 16
    jl .edge_rows_check8
    
    ; 1x16 kernel
    vxorps zmm15, zmm15, zmm15
    mov rax, r8
    imul rax, rcx
    shl rax, 2
    lea rdi, [r14 + rax]
    xor rax, rax
.edge_rows_dot16_single:
    mov rdx, rax
    imul rdx, rbx
    shl rdx, 2
    add rdx, r13
    vmovups zmm0, [rdx + r9*4]
    vbroadcastss zmm2, [rdi]
    vfmadd231ps zmm15, zmm2, zmm0
    add rdi, 4
    inc rax
    cmp rax, rcx
    jl .edge_rows_dot16_single
    mov rax, r8
    imul rax, rbx
    add rax, r9
    shl rax, 2
    add rax, r15
    vmovups [rax], zmm15
    add r9, 16
    jmp .edge_rows_col_loop

.edge_rows_check8:
    mov rax, rbx
    sub rax, r9
    cmp rax, 8
    jl .edge_rows_scalar
    
    ; 1x8 kernel (use ymm for 8 floats)
    vxorps ymm15, ymm15, ymm15
    mov rax, r8
    imul rax, rcx
    shl rax, 2
    lea rdi, [r14 + rax]
    xor rax, rax
.edge_rows_dot8:
    mov rdx, rax
    imul rdx, rbx
    shl rdx, 2
    add rdx, r13
    vmovups ymm0, [rdx + r9*4]
    vbroadcastss ymm2, [rdi]
    vfmadd231ps ymm15, ymm2, ymm0
    add rdi, 4
    inc rax
    cmp rax, rcx
    jl .edge_rows_dot8
    mov rax, r8
    imul rax, rbx
    add rax, r9
    shl rax, 2
    add rax, r15
    vmovups [rax], ymm15
    add r9, 8
    jmp .edge_rows_col_loop
    
.edge_rows_scalar:
    vxorps xmm15, xmm15, xmm15
    mov rax, r8
    imul rax, rcx
    shl rax, 2
    lea rdi, [r14 + rax]
    xor rax, rax
.edge_rows_dot1:
    mov rdx, rax
    imul rdx, rbx
    add rdx, r9
    shl rdx, 2
    add rdx, r13
    vmovss xmm0, [rdx]
    vmovss xmm2, [rdi]
    vfmadd231ss xmm15, xmm2, xmm0
    add rdi, 4
    inc rax
    cmp rax, rcx
    jl .edge_rows_dot1
    mov rax, r8
    imul rax, rbx
    add rax, r9
    shl rax, 2
    add rax, r15
    vmovss [rax], xmm15
    inc r9
    jmp .edge_rows_col_loop
    
.edge_rows_next:
    inc r8
    jmp .edge_rows_loop

.done:
    add rsp, 56
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
