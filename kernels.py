"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement dsd_matmul (A1)")


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement sparse_flash_forward (A2)")


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement sparse_flash_backward (A3)")
