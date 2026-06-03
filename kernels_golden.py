"""Golden-serving stub for SPARSE_REF=1 smoke-tests. GIVEN -- do not edit.

When harness.py imports this module in place of `kernels`, the three launch
wrappers become lookups into golden.pt -- a frozen set of reference outputs
produced offline by the instructor's reference kernels. This lets you verify
your environment and the harness wiring end-to-end before you have any working
kernels, WITHOUT shipping the reference kernel source.

What this is NOT: a working reference implementation. The wrappers here compute
nothing; they copy precomputed reference outputs into freshly allocated tensors.
Use SPARSE_REF=1 only as a "does my environment work" smoke-test.

Keyed by the sanity_check.py cases:
    ('A1', M, K, N, block)            -> {'C'}
    ('A2', B, H, T, d, block)         -> {'O', 'L'}
    ('A3', B, H, T, d, block)         -> {'dQ', 'dK', 'dV'}
"""
import os
import warnings

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_PATH = os.path.join(_HERE, "golden.pt")

try:
    _GOLDENS = torch.load(_GOLDEN_PATH, map_location="cpu", weights_only=False)
except FileNotFoundError:
    _GOLDENS = None


def _lookup(key):
    if _GOLDENS is None:
        raise FileNotFoundError(
            f"{_GOLDEN_PATH} not found. SPARSE_REF=1 needs the precomputed golden "
            f"bundle. Run with SPARSE_REF=0 (your own kernels) or obtain golden.pt "
            f"from the course materials."
        )
    if key not in _GOLDENS:
        have = sorted(_GOLDENS.keys(), key=str)
        raise KeyError(
            f"no golden entry for {key}. SPARSE_REF=1 only covers the default "
            f"sanity_check cases ({have}). If you changed SEED or the cases, run "
            f"with SPARSE_REF=0 and compare against the fp64 reference live."
        )
    return _GOLDENS[key]


def _check_fp(key, fp):
    """Warn (non-fatal) if the live inputs differ from the ones the golden was
    computed on -- a PASS here would otherwise be meaningless."""
    stored = _GOLDENS[key].get("fp")
    if stored is not None and abs(float(fp) - float(stored)) > 1e-2:
        warnings.warn(
            f"SPARSE_REF=1: input fingerprint for {key} differs from the stored "
            f"one ({fp:.3e} vs {stored:.3e}). The served output was computed on a "
            f"different input; PASS here does not mean your kernel is right.",
            stacklevel=3,
        )


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    key = ("A1", int(M), int(K), int(N), int(block))
    entry = _lookup(key)
    _check_fp(key, float(values.float().sum()) + float(B.float().sum()))
    return entry["C"].to(B.device, dtype=torch.float32)


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    Bb, H, T, d = Q.shape
    key = ("A2", int(Bb), int(H), int(T), int(d), int(BLOCK_Q))
    entry = _lookup(key)
    _check_fp(key, float(Q.float().sum()) + float(K.float().sum()) + float(V.float().sum()))
    O = entry["O"].to(Q.device, dtype=Q.dtype)
    L = entry["L"].to(Q.device, dtype=torch.float32)
    return O, L


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,
                          q_row_offsets, q_col_indices,
                          sm_scale, BLOCK_Q, BLOCK_K):
    Bb, H, T, d = Q.shape
    key = ("A3", int(Bb), int(H), int(T), int(d), int(BLOCK_Q))
    entry = _lookup(key)
    _check_fp(key, float(Q.float().sum()) + float(dO.float().sum()))
    dQ = entry["dQ"].to(Q.device, dtype=Q.dtype)
    dK = entry["dK"].to(Q.device, dtype=Q.dtype)
    dV = entry["dV"].to(Q.device, dtype=Q.dtype)
    return dQ, dK, dV
