# Algorithms: block-sparse attention

The spec for the three functions in `kernels.py`.
You implement them from this document plus their given signatures
(the input and output shapes and dtypes). This document gives you the math, which
is what each output is, and the layout of the given inputs.
Implementation details (number of kernels, the grid, the loops, and the tiling) are left open.

## 0. Notation and Tensor Shapes

- `Q, K, V, O, dO` are `(B, H, T, d)`. Attention is independent per `(b, h)`, so
  the `B·H` heads are separate problems. `sm_scale = 1/sqrt(d)`, written `σ`.
- The block size is `block`, square, so `BLOCK_Q = BLOCK_K = block`. `T` is a
  multiple of `block`, and `n = T / block` is the number of blocks per axis.
- `log2(x)` and `exp2(x) = 2^x`, with `LOG2E = log2(e) ≈ 1.4427`. The residual
  `L` (below) is stored in log2 units, so the math is written in base 2.

## 0.1 What is given (shapes and dtypes)

You implement the three functions in `kernels.py`. The only thing fixed is their
signatures: the shapes and dtypes of the inputs and outputs. The grader asserts
the returned shapes and dtypes, then checks values against an fp64 reference.

```
dsd_matmul(values (nnz,block,block) f32, row_offsets (n+1,) i32,
           column_indices (nnz,) i32, B (K,N) f32, M,K,N,block) -> C (M,N) f32

sparse_flash_forward(Q,K,V (B,H,T,d) f16, q_row_offsets (n+1,) i32,
           q_col_indices (nnz,) i32, sm_scale, BLOCK_Q, BLOCK_K)
           -> O (B,H,T,d) f16, L (B,H,T) f32

sparse_flash_backward(Q,K,V,O (B,H,T,d) f16, L (B,H,T) f32, dO (B,H,T,d) f16,
           k_row_offsets, k_col_indices,   # the key-block view  (§1)
           q_row_offsets, q_col_indices,   # the query-block view (§1)
           sm_scale, BLOCK_Q, BLOCK_K)
           -> dQ, dK, dV (B,H,T,d) f16
```

- `BLOCK_Q == BLOCK_K == block` is the BCSR block granularity.
- The BCSR views are given inputs, built host-side by `bcsr.attn_bcsr_views`. You
  will never construct BCSR. They are `int32`, and §1 defines each view's layout.
- Store `O`, `dQ`, `dK`, `dV` in fp16 and `L` in fp32 (this is grader-enforced).
- Mask out-of-range rows and columns (`offs >= T`) on load (`other=0`), and set
  out-of-range scores to `-inf` before the softmax so dead keys vanish.
- §2 to §4 give the math each function must produce. How you decompose it into
  kernels and place it on the grid is yours.

## 1. BCSR and the two transpose views

A boolean block mask `M ∈ {0,1}^(n×n)` says which (query block, key block) pairs
interact. BCSR packs the live blocks of a mask, row-major:

```
row_offsets     (n+1,)   prefix sum of live blocks per block-row
column_indices  (nnz,)   block-column of each live block, row-major
```

Block-row `r` owns the live blocks `column_indices[row_offsets[r] :
row_offsets[r+1]]`. That slice is the set of block-columns live in row `r`.

The backward is given both transpose views of the one mask, built host-side by
`bcsr.attn_bcsr_views`, off the kernel clock:

- query-block view `(q_row_offsets, q_col_indices)`: the rows of `M`. For query
  block `i`, the key blocks `j` it attends.
- key-block view `(k_row_offsets, k_col_indices)`: the rows of `Mᵀ`. For key
  block `j`, the query blocks `i` that attend it.

Every query block-row is guaranteed at least one live block, so the softmax
denominator is always defined.

## 2. A1: DSD, block-sparse `A @ B`

`A` is `(M, K)`, block-sparse in BCSR (§1). `B` is dense `(K, N)`. `C = A·B` is
dense `(M, N)`. Because `A`'s dead blocks are zero, each output tile is the sum
over only `A`'s live K-blocks:

```
C[i-tile, :] = Σ over k live in block-row i  of  A_block(i,k) · B[k·block:(k+1)·block, :]
```

A block-row with `F` live K-blocks does `F` block-matmuls instead of `K/block`,
so the saving is the zero fraction of `A`'s blocks. fp32 throughout,
`allow_tf32=False`.

## 3. A2: sparse flash forward

For each query block `i`, attention is taken over only its live key blocks (the
query-block view, §1), with scores masked so out-of-range and off-pattern keys
contribute nothing:

```
O_i = softmax_j( σ · Q_i · K_jᵀ ) · V_j      (j over the live key blocks of row i)
```

You also return the per-row residual

```
L_i = LOG2E · logsumexp_j( σ · Q_i · K_j )    (over the same live key blocks)
```

`L` is one fp32 scalar per query row, in log2 units. It is the only forward
quantity the backward reuses.

## 4. A3: sparse flash backward

Given `O`, `L` from the forward and the upstream grad `dO`, the attention
probabilities are recovered from the stored residual,

```
P_ij = exp2( σ · LOG2E · Q_i·K_j − L_i )      (= softmax_j over row i's live blocks)
```

and the gradients are

```
(1)  D_i   = Σ_d dO_id · O_id                  # = rowsum(dO ⊙ O)
(2)  dV_j  = Σ_i P_ij · dO_i
(3)  dP_ij = dO_i · V_j
(4)  dS_ij = P_ij · (dP_ij − D_i)
(5)  dQ_i  = σ · Σ_j dS_ij · K_j
     dK_j  = σ · Σ_i dS_ij · Q_i
```