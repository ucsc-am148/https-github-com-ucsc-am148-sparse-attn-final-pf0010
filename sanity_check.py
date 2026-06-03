"""Local sanity check for the block-sparse attention assignment.

Runs each rung (A1 DSD, A2 forward, A3 backward) at a couple of fixed sizes,
compares against an fp64 ground-truth reference, and prints max relative error +
a rough kernel time + PASS/FAIL.

This is NOT the autograder. It checks correctness only; the grader runs a wider
battery, enforces a loose anti-cheese perf floor, and times the leaderboard step.

Usage:
    python sanity_check.py             # run your kernels.py
    SPARSE_REF=1 python sanity_check.py  # serve precomputed reference outputs (smoke-test)

Until you implement a rung's function, it raises `NotImplementedError`, so that
rung FAILs.
"""
import time

import torch

import harness


SEED = 0

# DSD: square and rectangular shapes. block divides M and K.
DSD_CASES = [
    {"M": 512,  "K": 512,  "N": 512,  "block": 128, "density": 0.40},
    {"M": 1024, "K": 768,  "N": 1024, "block": 128, "density": 0.50},
    {"M": 768,  "K": 1024, "N": 512,  "block": 256, "density": 0.30},
]

# Attention: a random-mask case and a causal-mask case.
ATTN_CASES = [
    {"B": 2, "H": 4, "T": 512,  "d": 64, "block": 64, "density": 0.25, "causal": False},
    {"B": 1, "H": 2, "T": 1024, "d": 64, "block": 64, "density": 1.00, "causal": True},
]

TOL_DSD = 1e-2
TOL_ATTN = 2e-2


def _rel(y, y_ref):
    y, y_ref = y.double(), y_ref.double()
    denom = y_ref.abs().max().clamp_min(1e-12)
    return ((y - y_ref).abs().max() / denom).item()


def _bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _row(name, shape, err, tol, ms):
    verdict = "PASS" if (err is not None and err < tol) else "FAIL"
    errs = f"{err:.3e}" if err is not None else "   --   "
    mss = f"{ms:8.3f}" if ms is not None else "    --  "
    print(f"  {name:>3}  {shape:<24}  {errs:>12}  {mss}   {verdict}  (tol {tol:.0e})")


def run_a1(case, device):
    plan = harness.a1_prepare(seed=SEED, device=device, **case)
    shape = f"{case['M']}x{case['N']}x{case['K']} blk{case['block']} d{plan['density']:.2f}"
    try:
        C = harness.a1_run(plan)
        ref = harness.a1_reference(plan)
        err = _rel(C, ref)
        ms = _bench(lambda: harness.a1_run(plan))
    except Exception as e:
        print(f"  A1  {shape:<24}   FAIL  ({type(e).__name__}: {e})")
        return
    _row("A1", shape, err, TOL_DSD, ms)


def run_a2_a3(case, device):
    plan = harness.attn_prepare(seed=SEED, device=device, **case)
    shape = (f"B{case['B']}H{case['H']}T{case['T']}d{case['d']} "
             f"blk{case['block']} d{plan['density']:.2f}"
             f"{' caus' if case['causal'] else ''}")
    inp = harness.attn_inputs(plan)
    ref = harness.attn_reference(inp, plan)
    O_ref16 = ref["O"].to(torch.float16)
    L_ref32 = ref["L"].to(torch.float32)

    # A2 forward
    try:
        O, L = harness.a2_run(inp, plan)
        err_o = _rel(O, ref["O"])
        err_l = _rel(L, ref["L"])
        ms2 = _bench(lambda: harness.a2_run(inp, plan))
        _row("A2", shape + " [O]", err_o, TOL_ATTN, ms2)
        _row("A2", shape + " [L]", err_l, TOL_ATTN, None)
    except Exception as e:
        print(f"  A2  {shape:<24}   FAIL  ({type(e).__name__}: {e})")

    # A3 backward (fed reference O, L so it is graded independent of A2)
    try:
        dQ, dK, dV = harness.a3_run(inp, O_ref16, L_ref32, plan)
        err = max(_rel(dQ, ref["dQ"]), _rel(dK, ref["dK"]), _rel(dV, ref["dV"]))
        ms3 = _bench(lambda: harness.a3_run(inp, O_ref16, L_ref32, plan))
        _row("A3", shape + " [dQdKdV]", err, TOL_ATTN, ms3)
    except Exception as e:
        print(f"  A3  {shape:<24}   FAIL  ({type(e).__name__}: {e})")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; cannot run the kernels.")
        return
    device = "cuda"
    print(f"device:   {torch.cuda.get_device_name(0)}")
    print(f"torch:    {torch.__version__}")
    try:
        import triton
        print(f"triton:   {triton.__version__}")
    except ImportError:
        print("triton:   not installed")
    print(f"SPARSE_REF: {'1 (golden stub)' if harness.USE_REF else '0 (your kernels)'}")
    print()
    print(f"  {'':>3}  {'shape':<24}  {'max rel err':>12}  {'ms':>8}   verdict")
    print(f"  {'-'*3}  {'-'*24}  {'-'*12}  {'-'*8}   {'-'*7}")
    for case in DSD_CASES:
        run_a1(case, device)
    for case in ATTN_CASES:
        run_a2_a3(case, device)


if __name__ == "__main__":
    main()
