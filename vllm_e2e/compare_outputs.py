"""Compare two smoke_infer.py dumps: token exact-match + logprob divergence.

    python compare_outputs.py results/smoke_native.json results/smoke_fi_dg.json
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    print(f"A: {a['tag']} (fi={a['fi_moe_ep']}/{a['fi_megakernel']})")
    print(f"B: {b['tag']} (fi={b['fi_moe_ep']}/{b['fi_megakernel']})")

    n_exact = 0
    for i, (ra, rb) in enumerate(zip(a["records"], b["records"])):
        ta, tb = ra["token_ids"], rb["token_ids"]
        common = min(len(ta), len(tb))
        div = next((j for j in range(common) if ta[j] != tb[j]), None)
        exact = div is None and len(ta) == len(tb)
        n_exact += exact
        pref = common if div is None else div
        lp_deltas = [
            abs(x - y)
            for x, y in zip(ra["logprobs"][:pref], rb["logprobs"][:pref])
            if x is not None and y is not None
        ]
        mean_dlp = sum(lp_deltas) / len(lp_deltas) if lp_deltas else float("nan")
        max_dlp = max(lp_deltas) if lp_deltas else float("nan")
        status = "EXACT" if exact else f"diverge@{div}"
        print(
            f"  [{i}] {status:>12}  matched_prefix={pref:3d}  "
            f"mean|dlogprob|={mean_dlp:.4f}  max={max_dlp:.4f}  "
            f"A={ra['text'][:40]!r} B={rb['text'][:40]!r}"
        )
    print(f"exact-match: {n_exact}/{len(a['records'])}")


if __name__ == "__main__":
    main()
