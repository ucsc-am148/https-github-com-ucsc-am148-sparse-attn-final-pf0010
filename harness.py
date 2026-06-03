"""Harness for the block-sparse attention ladder A1 / A2 / A3. GIVEN -- do not edit.

Wires your kernels.py into prepare / inputs / run helpers the sanity check and
the autograder both call, and provides the fp64 ground-truth references the
grader scores against.

Module toggle: set SPARSE_REF=1 in the environment to swap in kernels_golden
(a stub that serves precomputed reference outputs from golden.pt). Useful for
verifying the harness wiring + your environment end-to-end before you have
working kernels; NOT a working reference kernel.

Layout convention for the kernels: Q, K, V, O, dO are (B, H, T, d); the
launchers flatten the leading (B, H) into one batch axis. L (log2 of the
softmax denominator) is (B, H, T). The block mask is square (T/block,
T/block); bcsr.attn_bcsr_views turns it into the two transpose views the
forward and the two backward kernels walk.
"""
import math
import os

import torch

import bcsr

# ---- module toggle: student kernels vs golden-serving stub ----
USE_REF = os.environ.get("SPARSE_REF", "0") == "1"
if USE_REF:
    import kernels_golden as kernels
else:
    import kernels


LOG2E = 1.4426950408889634


# =============================================================================
# A1 -- DSD: block-sparse (BCSR) @ dense
# =============================================================================

def a1_prepare(M, K, N, block, density, seed, device="cuda"):
    """Build the A1 problem off the kernel clock: a block-sparse (M, K) tensor
    packed to BCSR, and a dense (K, N) right operand. `A_dense` is the densified
    BCSR -- exactly the operand the kernel sees -- so the reference matches the
    live blocks bit-for-bit (off-pattern blocks are zero, not the original A)."""
    A, mask = bcsr.make_block_sparse(M, K, block, density, seed=seed, device=device)
    values, row_offsets, column_indices = bcsr.to_bcsr(A, block, mask)
    g = torch.Generator(device=device).manual_seed(seed + 100)
    Bmat = torch.randn(K, N, generator=g, device=device, dtype=torch.float32)
    nbr, nbc = M // block, K // block
    A_dense = bcsr.densify(values, row_offsets, column_indices, nbr, nbc, block)
    return {
        "values": values, "row_offsets": row_offsets, "column_indices": column_indices,
        "B": Bmat, "M": M, "K": K, "N": N, "block": block,
        "density": mask.float().mean().item(), "nnz": int(values.shape[0]),
        "A_dense": A_dense, "device": device,
    }


def a1_run(plan):
    """Run the student DSD kernel -> dense (M, N)."""
    return kernels.dsd_matmul(
        plan["values"], plan["row_offsets"], plan["column_indices"],
        plan["B"], plan["M"], plan["K"], plan["N"], plan["block"],
    )


def a1_reference(plan):
    """Ground truth: densified sparse A @ B, fp64 accumulation, cast to fp32."""
    return (plan["A_dense"].double() @ plan["B"].double()).to(torch.float32)


# =============================================================================
# A2 / A3 -- block-sparse attention (shared problem)
# =============================================================================

def attn_prepare(B, H, T, d, block, density, seed, causal=False, device="cuda"):
    """Build the attention problem off the kernel clock: a square block mask
    (random at `density`, or causal), both BCSR transpose views, and sm_scale.
    Every query block-row is guaranteed at least one live block so softmax is
    defined."""
    nb = T // block
    if causal:
        mask = bcsr.causal_block_mask(nb, device=device)
    else:
        mask = bcsr.random_block_mask(nb, density, seed=seed, device=device)
    mask = bcsr.ensure_live_query_rows(mask)
    qro, qci, kro, kci = bcsr.attn_bcsr_views(mask)
    return {
        "B": B, "H": H, "T": T, "d": d, "block": block,
        "mask": mask,
        "q_row_offsets": qro, "q_col_indices": qci,   # query-block view (fwd, dQ)
        "k_row_offsets": kro, "k_col_indices": kci,   # key-block view  (dK/dV)
        "sm_scale": 1.0 / (d ** 0.5),
        "density": mask.float().mean().item(), "nnz": int(mask.sum().item()),
        "seed": seed, "device": device,
    }


def attn_inputs(plan, dtype=torch.float16):
    """Deterministic Q, K, V, dO for this plan. Same plan seed -> same inputs
    across sanity_check, the grader, and build_goldens."""
    B, H, T, d = plan["B"], plan["H"], plan["T"], plan["d"]
    g = torch.Generator(device=plan["device"]).manual_seed(plan["seed"] + 1000)

    def r():
        return torch.randn(B, H, T, d, generator=g, device=plan["device"], dtype=dtype)

    return {"Q": r(), "K": r(), "V": r(), "dO": r()}


def a2_run(inp, plan):
    """Run the student forward -> (O (B,H,T,d), L (B,H,T))."""
    return kernels.sparse_flash_forward(
        inp["Q"], inp["K"], inp["V"],
        plan["q_row_offsets"], plan["q_col_indices"],
        plan["sm_scale"], plan["block"], plan["block"],
    )


def a3_run(inp, O, L, plan):
    """Run the student backward -> (dQ, dK, dV). Consumes the forward residuals
    O, L. The grader feeds reference O, L so A3 is graded independent of A2."""
    return kernels.sparse_flash_backward(
        inp["Q"], inp["K"], inp["V"], O, L, inp["dO"],
        plan["k_row_offsets"], plan["k_col_indices"],
        plan["q_row_offsets"], plan["q_col_indices"],
        plan["sm_scale"], plan["block"], plan["block"],
    )


def step_run(inp, plan):
    """One full training step: forward (produces O, L) then backward (consumes
    O, L, dO -> dQ, dK, dV), back to back. This is the leaderboard's timed
    region. Returns (O, L, dQ, dK, dV)."""
    O, L = a2_run(inp, plan)
    dQ, dK, dV = a3_run(inp, O, L, plan)
    return O, L, dQ, dK, dV


def attn_reference(inp, plan):
    """fp64 ground truth for O, L, dQ, dK, dV via materialized masked attention.

    Expands the block mask to a (T, T) element mask, runs softmax with the
    masked scores, and backprops the given dO. L is log2 of the softmax
    denominator (LOG2E * logsumexp over the masked, scaled scores), matching
    the exp2 convention the kernels store.
    """
    B, H, T, d = plan["B"], plan["H"], plan["T"], plan["d"]
    block, sm_scale = plan["block"], plan["sm_scale"]
    elem = plan["mask"].repeat_interleave(block, 0).repeat_interleave(block, 1).bool()

    Qd = inp["Q"].double().detach().requires_grad_()
    Kd = inp["K"].double().detach().requires_grad_()
    Vd = inp["V"].double().detach().requires_grad_()

    S = (Qd @ Kd.transpose(-1, -2)) * sm_scale
    S = S.masked_fill(~elem, float("-inf"))
    P = torch.softmax(S, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    O = P @ Vd
    O.backward(inp["dO"].double())

    L = LOG2E * torch.logsumexp(S, dim=-1)   # (B, H, T); live rows are finite
    return {
        "O": O.detach(), "L": L.detach(),
        "dQ": Qd.grad, "dK": Kd.grad, "dV": Vd.grad,
    }


# =============================================================================
# Leaderboard FLOP accounting
# =============================================================================

def step_useful_flops(plan):
    """Useful FLOPs for one forward+backward step, counting ONLY live block
    pairs (densifying earns no credit). Per live (query, key) block pair and per
    (batch, head): forward ~= 4*BQ*BK*d (QK^T, P@V), backward ~= 10*BQ*BK*d
    (recompute S, dV, dP, dK, dQ -- recompute counted once). Step total
    = 14*BQ*BK*d (a ~2.5x backward:forward ratio).
    """
    BQ = BK = plan["block"]
    per_pair = 14.0 * BQ * BK * plan["d"]
    return plan["B"] * plan["H"] * plan["nnz"] * per_pair
