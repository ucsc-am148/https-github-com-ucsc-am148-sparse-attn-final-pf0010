"""Host-side block-sparse (BCSR) helpers. GIVEN -- do not edit.

The DSD packing serves A1; the dual transpose-view builder (attn_bcsr_views)
serves the A2/A3 backward.

BCSR is block-CSR: take CSR and make every stored entry a dense block-by-block
tile instead of a scalar.

  row_offsets     (n_block_rows + 1,)  int32   prefix sum of nonzero blocks per
                                                block-row (a CSR-style offsets
                                                vector, now over blocks).
  column_indices  (nnz_blocks,)        int32   block-column of each nonzero
                                                block, in row-major order.
  values          (nnz_blocks, B, B)   float32 the packed nonzero block data
                                                (DSD only -- attention's Q/K/V
                                                are dense, so the attention path
                                                stores only offsets + indices).

A block-row r owns the nonzero blocks values[row_offsets[r] : row_offsets[r+1]]
sitting at columns column_indices[row_offsets[r] : row_offsets[r+1]]. The kernels
walk that slice as their K-loop; the dense kernel walked all of K.

Two uses in this assignment:
  * A1 (DSD): pack a block-sparse (M, K) dense tensor with `to_bcsr`. The
    rectangular (M/block, K/block) mask drives a single sparse K-loop.
  * A2/A3 (attention): the square (T/block, T/block) block mask drives the
    sparse forward and the two backward kernels. The backward needs BOTH
    transpose views of the one mask -- see `attn_bcsr_views`.
"""
import torch


# =============================================================================
# Block masks
# =============================================================================

def block_sparse_mask(n_block_rows, n_block_cols, density, seed=0, device="cuda"):
    """Boolean (n_block_rows, n_block_cols) mask, each block live i.i.d. w.p. density."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(n_block_rows, n_block_cols, generator=g, device=device) < density


def random_block_mask(n_blocks, density, seed=0, device="cuda"):
    """Square (n_blocks, n_blocks) random block mask for the attention rungs."""
    return block_sparse_mask(n_blocks, n_blocks, density, seed=seed, device=device)


def causal_block_mask(n_blocks, device="cuda"):
    """Lower-triangular square block mask (query block i attends key block j<=i)."""
    return torch.tril(torch.ones(n_blocks, n_blocks, device=device, dtype=torch.bool))


def ensure_live_query_rows(mask):
    """Guarantee every query block-row (row of `mask`) has >= 1 live block.

    A query row with no live key block has an undefined softmax (all -inf
    scores). The forward zeroes those rows and the masked-attention reference
    NaN-fixes them, but to keep the correctness battery and the leaderboard
    well-posed we put a live block on the diagonal of any empty row. Returns the
    same tensor (mutated in place) for convenience.
    """
    no_live = mask.sum(dim=1) == 0
    if no_live.any():
        idx = no_live.nonzero(as_tuple=False).flatten()
        mask[idx, idx] = True
    return mask


# =============================================================================
# A1 (DSD): pack a block-sparse dense tensor
# =============================================================================

def make_block_sparse(M, K, block, density, seed=0, device="cuda"):
    """A dense (M, K) tensor whose block-by-block tiles are zeroed off-pattern.

    Returns (A, mask) where mask is the (M/block, K/block) block pattern. The
    measured density is mask.float().mean(); for small grids it drifts from the
    requested density.
    """
    assert M % block == 0 and K % block == 0
    nbr, nbc = M // block, K // block
    mask = block_sparse_mask(nbr, nbc, density, seed, device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    A = torch.randn(M, K, generator=g, device=device, dtype=torch.float32)
    full = mask.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return A * full, mask


def to_bcsr(A, block, mask=None):
    """Pack a block-sparse dense tensor into (values, row_offsets, column_indices).

    If mask is None the nonzero blocks are detected from A (a block is live if
    any element is nonzero). Block order is row-major, matching torch.nonzero.
    """
    M, K = A.shape
    assert M % block == 0 and K % block == 0
    nbr, nbc = M // block, K // block
    # (nbr, nbc, block, block): block (r, c) is blocks[r, c].
    blocks = A.view(nbr, block, nbc, block).permute(0, 2, 1, 3).contiguous()
    if mask is None:
        mask = blocks.abs().amax(dim=(2, 3)) > 0

    row_offsets, column_indices = mask_to_offsets(mask)
    values = blocks[mask].contiguous()  # (nnz, block, block), nonzero (row-major) order
    return values, row_offsets, column_indices


def densify(values, row_offsets, column_indices, nbr, nbc, block):
    """Inverse of to_bcsr: scatter the packed blocks back to a dense (M, K)."""
    out = torch.zeros(nbr, nbc, block, block, device=values.device, dtype=values.dtype)
    rows = torch.repeat_interleave(
        torch.arange(nbr, device=values.device),
        (row_offsets[1:] - row_offsets[:-1]).to(torch.int64),
    )
    out[rows, column_indices.to(torch.int64)] = values
    return out.permute(0, 2, 1, 3).reshape(nbr * block, nbc * block)


def validate_bcsr(A, block):
    """Round-trip check: to_bcsr then densify reproduces A on the live blocks."""
    nbr, nbc = A.shape[0] // block, A.shape[1] // block
    values, row_offsets, column_indices = to_bcsr(A, block)
    back = densify(values, row_offsets, column_indices, nbr, nbc, block)
    return (back - A).abs().max().item()


# =============================================================================
# Shared: mask -> (row_offsets, column_indices)
# =============================================================================

def mask_to_offsets(mask):
    """Vectorized BCSR offsets/indices for any boolean (nbr, nbc) block mask.

    Returns (row_offsets (nbr+1,) int32, column_indices (nnz,) int32) in
    row-major (torch.nonzero) order -- the same order to_bcsr packs `values`.
    """
    nbr, _ = mask.shape
    rows, cols = torch.nonzero(mask, as_tuple=True)  # row-major
    column_indices = cols.to(torch.int32)
    counts = torch.bincount(rows, minlength=nbr)
    row_offsets = torch.zeros(nbr + 1, device=mask.device, dtype=torch.int32)
    row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return row_offsets, column_indices


# =============================================================================
# A2 / A3 (attention): both transpose views from one mask
# =============================================================================

def attn_bcsr_views(mask):
    """Build both BCSR views the sparse forward + backward need from one mask.

    The forward and dQ kernel iterate, for each query block i, the key blocks j
    it attends (the rows of `mask`). The dK/dV kernel iterates, for each key
    block j, the query blocks i that attend it (the rows of `mask.T`).

    Returns a 4-tuple:
        q_row_offsets, q_col_indices   query-block view  (rows of mask):
                                       for query block i -> its key blocks j.
                                       Used by forward and the dQ kernel.
        k_row_offsets, k_col_indices   key-block view    (rows of mask.T):
                                       for key block j -> the query blocks i.
                                       Used by the dK/dV kernel.

    Naming convention: each view is named by the block axis that indexes its
    rows. The `q_*` arrays are the query-block view (rows indexed by query block
    i, listing the key blocks it attends, = rows of mask); the `k_*` arrays are
    the key-block view (rows indexed by key block j, listing the query blocks
    that attend it, = rows of mask.T). Built host-side, off the kernel clock.
    """
    assert mask.shape[0] == mask.shape[1], "attention block mask must be square"
    q_row_offsets, q_col_indices = mask_to_offsets(mask)                 # query-block view
    k_row_offsets, k_col_indices = mask_to_offsets(mask.t().contiguous())  # key-block view
    return q_row_offsets, q_col_indices, k_row_offsets, k_col_indices


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # A1 round-trip
    for (M, K, block, density) in [(512, 512, 128, 0.3), (1024, 768, 128, 0.5), (2048, 2048, 256, 0.1)]:
        A, mask = make_block_sparse(M, K, block, density, seed=0, device=dev)
        values, row_offsets, column_indices = to_bcsr(A, block, mask)
        nbr, nbc = M // block, K // block
        nnz = values.shape[0]
        err = validate_bcsr(A, block)
        print(f"DSD {M}x{K} block={block} req={density}: nnz={nnz}/{nbr * nbc} "
              f"realized={nnz / (nbr * nbc):.3f} roundtrip_err={err:.4f}")
        assert err == 0.0, "BCSR round-trip must be exact"

    # A2/A3 dual view consistency: the two views must agree on the live set.
    m = ensure_live_query_rows(random_block_mask(8, 0.25, seed=7, device=dev))
    kro, kci, qro, qci = attn_bcsr_views(m)
    assert int(kro[-1]) == int(qro[-1]) == int(m.sum()), "views disagree on nnz"
    print(f"attn dual-view: nnz={int(m.sum())}/{m.numel()} consistent")
