#!/bin/bash
# Quantized combine-wire experiment (fi_nvfp4): tune wire-valid candidates at
# prefill shapes, then bench prefill-eager + decode-graphs with the nvfp4
# wire, plus a correctness smoke (wire quantizes combine partials — expect a
# slightly larger |dlogprob| band; microbench acc_loss 25.0% vs 23.2% bf16).
# Baselines: prefill-eager 28204 (bf16 wire), decode-graphs 10413 (dectuned).
# Markers: logs/combine_wire.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%H%M%S)
CACHE_W=$W/results/knob_cache_dsv4_wire_nvfp4.json
FIENV="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FI_MOE_EP_IKR=0 FI_MOE_EP_COMBINE=nvfp4"
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
echo "=== tune (combine-dtype nvfp4, wire-valid candidates, bucket 4096) ==="
rm -f "$CACHE_W"
run "source venv0251/bin/activate && \
  FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE_W \
  torchrun --standalone --nproc_per_node=4 -m flashinfer.moe_ep.tune \
    --dtype nvfp4 --hidden 4096 --intermediate 2048 \
    --num-experts 256 --topk 6 --max-tokens 4096 --combine-dtype nvfp4" \
  > "$W/logs/combine_wire_tune.log" 2>&1 || rc=1
grep -E "winner" "$W/logs/combine_wire_tune.log" | tail -1

echo "=== smoke (combine_nvfp4 correctness) ==="
run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
  env $FIENV FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE_W \
  python smoke_infer.py --tag fi_wire --out results/smoke_wire_$STAMP.json && \
  python compare_outputs.py results/smoke_native.json results/smoke_wire_$STAMP.json" \
  > "$W/logs/combine_wire_smoke.log" 2>&1 || rc=1
grep -E "dlogprob|exact" "$W/logs/combine_wire_smoke.log" | tail -3

for cell in "prefill_e:ENFORCE_EAGER=1:prefill:1024:1" "decode_g:ENFORCE_EAGER=0:decode:128:256"; do
  IFS=: read -r name eager wname ilen olen <<< "$cell"
  echo "=== bench wire $name ==="
  run "source venv0251/bin/activate && \
    env $FIENV FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE_W $eager \
    python bench_offline.py --tag wire_$name --workload $wname:$ilen:$olen \
      --rounds 5 --out results/wire_${STAMP}_${name}.json" \
    > "$W/logs/combine_wire_${name}.log" 2>&1 || rc=1
  grep -E "median" "$W/logs/combine_wire_${name}.log" | tail -1
done
echo "$rc" > "$W/logs/combine_wire.done"
exit $rc
