"""GSM8K accuracy oracle for the moe_ep e2e backends (optional gate).

Boots ONE engine (env-selected backend + per-backend checkpoint, exactly like
bench_offline.py), runs few-shot greedy GSM8K, extracts the final number, and
reports accuracy. Purpose: the two-checkpoint policy (native/fi_dg on the
mx-format original, fi_nvfp4 on the NVFP4 cast of the same base weights) makes
throughput comparisons cross-checkpoint — this oracle is the fairness gate.
Both paths must score in the same band (expected ~0.95 for DSV4-Flash) before
a perf number is an apples-to-apples claim.

    # native / fi_dg (mx checkpoint):
    python eval_gsm8k.py --tag native --out results/gsm8k_native.json

    # fi_nvfp4 (NVFP4 checkpoint, resolved automatically):
    FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl \
    python eval_gsm8k.py --tag fi_nvfp4 --out results/gsm8k_fi_nvfp4.json

    # gate a CI-style run (exit 2 below threshold):
    python eval_gsm8k.py --tag fi_nvfp4 --min-acc 0.93 --out ...

Data: the shared verl parquet prep by default (GSM8K_DIR env to override);
falls back to the OpenAI jsonl download (same source as vllm
tests/evals/gsm8k). Answer extraction matches that eval: last number in the
completion, commas stripped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from bench_offline import resolve_model

DEFAULT_GSM8K_DIR = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/dataset/core-models_verl/gsm8k/prep1"
)
INVALID = float("nan")

# Fixed few-shot block: the first five GSM8K TRAIN examples with worked
# solutions in the dataset's own style (final answer after "####"). Embedded
# so the eval does not depend on any particular train-split packaging.
FEW_SHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100/2 = $50. Betty's grandparents gave her 15 x 2 = $30. This means, Betty needs 100 - 50 - 30 - 15 = $5 more. #### 5

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Julie read 12 x 2 = 24 pages today. So she was able to read a total of 12 + 24 = 36 pages since yesterday. There are 120 - 36 = 84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages. #### 42

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 3 x 2 = 6 pages a week. So he writes 6 x 2 = 12 pages every week. That means he writes 12 x 52 = 624 pages a year. #### 624

"""


def extract_answer(text: str) -> float:
    """Last number in the text, commas stripped (vllm gsm8k_eval convention)."""
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return INVALID
    try:
        return float(numbers[-1])
    except ValueError:
        return INVALID


def _questions_from_parquet(path: str) -> list[tuple[str, float]]:
    """Read (question, ground_truth) pairs from a GSM8K parquet.

    Handles both the plain HF schema (question/answer columns) and the verl
    prep schema (prompt message list + reward_model.ground_truth).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = set(table.column_names)
    out: list[tuple[str, float]] = []
    if {"question", "answer"} <= cols:
        for q, a in zip(
            table.column("question").to_pylist(), table.column("answer").to_pylist()
        ):
            out.append((q, extract_answer(a)))
    elif {"prompt", "reward_model"} <= cols:
        for prompt, rm in zip(
            table.column("prompt").to_pylist(),
            table.column("reward_model").to_pylist(),
        ):
            # verl prompts wrap the raw question (often with a CoT instruction
            # suffix); use the user-message content verbatim as the question.
            content = prompt[0]["content"] if prompt else ""
            out.append((content, extract_answer(str(rm.get("ground_truth", "")))))
    else:
        raise ValueError(f"unrecognized gsm8k parquet schema: {sorted(cols)}")
    return out


def _questions_from_jsonl(path: str) -> list[tuple[str, float]]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append((d["question"], extract_answer(d["answer"])))
    return out


def load_questions(limit: int) -> list[tuple[str, float]]:
    gsm8k_dir = os.environ.get("GSM8K_DIR", DEFAULT_GSM8K_DIR)
    parquet = os.path.join(gsm8k_dir, "test.parquet")
    jsonl = os.path.join(gsm8k_dir, "test.jsonl")
    if os.path.exists(parquet):
        qs = _questions_from_parquet(parquet)
    elif os.path.exists(jsonl):
        qs = _questions_from_jsonl(jsonl)
    else:
        raise SystemExit(
            f"no test.parquet/test.jsonl under {gsm8k_dir}; set GSM8K_DIR "
            "(vllm tests/evals/gsm8k/gsm8k_eval.py documents the jsonl source)"
        )
    qs = [(q, gt) for q, gt in qs if gt == gt]  # drop rows with unparseable truth
    return qs[:limit] if limit else qs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-questions", type=int, default=200,
                    help="0 = full test split (1319)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-acc", type=float, default=None,
                    help="optional gate: exit 2 if accuracy falls below "
                         "(DSV4-Flash band is ~0.95)")
    ap.add_argument("--dump-preds", action="store_true")
    ap.add_argument("--model", default=None,
                    help="checkpoint path; default resolves per backend "
                         "(bench_offline.resolve_model)")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    ap.add_argument("--dp", type=int, default=int(os.environ.get("DP", "1")))
    ap.add_argument("--enforce-eager", action="store_true",
                    default=os.environ.get("ENFORCE_EAGER", "1") == "1")
    args = ap.parse_args()
    args.model = resolve_model(args.model)
    print(f"[eval_gsm8k] {args.tag}: model = {args.model}", flush=True)

    questions = load_questions(args.num_questions)
    prompts = [FEW_SHOT + f"Question: {q}\nAnswer:" for q, _ in questions]
    labels = [gt for _, gt in questions]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v4",
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        enable_expert_parallel=True,
        moe_backend="deep_gemm_mega_moe",
        max_model_len=4096,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype="fp8",
        block_size=256,
        # Accuracy oracle, not a perf bench: prefix caching stays ON so the
        # shared few-shot prefix is computed once (bench_offline disables it
        # because there it fakes throughput; it cannot fake accuracy).
    )

    sp = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, stop=["Question:"]
    )
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    preds = [extract_answer(o.outputs[0].text) for o in outs]
    invalid = sum(1 for p in preds if p != p)
    correct = sum(1 for p, gt in zip(preds, labels) if p == p and p == gt)
    acc = correct / len(labels)

    payload = {
        "tag": args.tag,
        "model": args.model,
        "num_questions": len(labels),
        "accuracy": acc,
        "correct": correct,
        "invalid": invalid,
        "elapsed_s": dt,
        "eager": args.enforce_eager,
        "fi_moe_ep": os.environ.get("FI_MOE_EP", "0"),
        "fi_megakernel": os.environ.get("FI_MOE_EP_MEGAKERNEL", "deep_gemm_mega"),
    }
    if args.dump_preds:
        payload["preds"] = [
            {"question": q, "expected": gt, "got": p if p == p else None}
            for (q, gt), p in zip(questions, preds)
        ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(
        f"[eval_gsm8k] {args.tag}: accuracy {acc:.4f} "
        f"({correct}/{len(labels)}, {invalid} unparseable) -> {args.out}",
        flush=True,
    )

    if args.min_acc is not None and acc < args.min_acc:
        print(
            f"[eval_gsm8k] FAIL: {acc:.4f} < --min-acc {args.min_acc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
