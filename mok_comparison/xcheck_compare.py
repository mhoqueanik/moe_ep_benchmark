"""Compare the fi and MoK cross-check outputs (same-input protocol).

Usage: python xcheck_compare.py <xcheck_dir> [world_size] [tolerance_pct]

Loads out_fi_rank{r}.pt and out_mok_rank{r}.pt and reports per-rank and
overall relative-L2 distance plus max abs difference. Bitwise equality is
not expected: both sides quantize weights/activations to MXFP8 with their
own kernels and accumulate in different orders — agreement within a few
percent rel-L2 (MXFP8 noise) is a pass; tolerance default 5%.
"""

import sys

import torch


def main() -> None:
    xdir = sys.argv[1]
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tol_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    diff_sq, ref_sq, worst_abs = 0.0, 0.0, 0.0
    for r in range(world):
        fi = torch.load(f"{xdir}/out_fi_rank{r}.pt").float()
        mok = torch.load(f"{xdir}/out_mok_rank{r}.pt").float()
        assert fi.shape == mok.shape, (fi.shape, mok.shape)
        d = fi - mok
        rel = (d.square().sum() / mok.square().sum().clamp_min(1e-30)).sqrt().item()
        cos = torch.nn.functional.cosine_similarity(
            fi.flatten(), mok.flatten(), dim=0
        ).item()
        print(
            f"rank {r}: rel-L2 {100 * rel:.3f}%  max|diff| {d.abs().max().item():.4f}  "
            f"cosine {cos:.6f}"
        )
        diff_sq += d.square().sum().item()
        ref_sq += mok.square().sum().item()
        worst_abs = max(worst_abs, d.abs().max().item())

    overall = 100.0 * (diff_sq / max(ref_sq, 1e-30)) ** 0.5
    verdict = "PASS" if overall <= tol_pct else "FAIL"
    print(f"overall: rel-L2 {overall:.3f}%  max|diff| {worst_abs:.4f}  -> {verdict} (tol {tol_pct}%)")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
