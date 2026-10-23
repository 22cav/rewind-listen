#!/usr/bin/env bash
# Every experiment in the report, then every table. Timed runs (the "paced" ones, without --fast) need an idle Mac on
# power with Low Power Mode off. Optional: LTTS = a LibriTTS-R test-clean folder (the WER clips), WPY = a Python with
# mlx-whisper (the Whisper cascade), VPY + VOICEBENCH = a Python with VoiceBench's requirements and a clone of it
# (the BBH and IFEval scores), STPY = a Python with onnxruntime and huggingface_hub (Smart Turn's own score).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${DATA:-data}; L=${LOGS:-logs}; PY=.venv/bin/python
mkdir -p "$D" "$L"
run() { $PY bench/run.py "$@"; }
cascade() { $PY bench/cascade.py "$@"; }
Q4=mlx-community/Qwen3-4B-4bit; LL=mlx-community/Llama-3.2-3B-Instruct-4bit

# data: VoiceBench dev split, two fixed subsets of it, latency and WER clips, Smart Turn, pauses, BBH, IFEval
$PY bench/data.py voicebench $D/vb_dev
for s in "vb200 0 200" "vb300 1 300"; do set -- $s; mkdir -p $D/$1; $PY -c "import os, random
f = sorted(os.listdir('$D/vb_dev/qa')); random.Random($2).shuffle(f)
[os.path.exists(p := '$D/$1/' + x) or os.symlink(os.path.abspath('$D/vb_dev/qa/' + x), p) for x in f[:$3]]"; done
$PY bench/data.py pack $D/pack ${LTTS:-}
$PY bench/data.py smartturn $D/smartturn
$PY bench/data.py pauses $D/e4
$PY bench/data.py bbh $D/bbh
$PY bench/data.py ifeval $D/ifeval

# exactness and revisions; window length and recognizer size; completion score; endpointing
run $D/pack/qa $D/vb200 --fast --parity --max_tokens 128 --out $L/e1_lcp.jsonl
run $D/pack/qa --fast --parity --fp32 --max_tokens 128 --out $L/e1_lcp_fp32.jsonl
# a pass every 0.5 s instead of at pauses
run $D/pack/qa $D/vb200 --fast --probe --step 0.5 --out $L/e1_lcp_0.5.jsonl
run $D/vb200 --fast --probe --parity --window 12 --out $L/e2_110m_w12.jsonl
run $D/vb200 --fast --probe --parity --asr mlx-community/parakeet-tdt-0.6b-v3 --out $L/e2_06b_w8.jsonl
[ -z "$(ls $D/pack/asr)" ] || run $D/pack/asr --fast --probe --parity --out $L/e2_libritts.jsonl
for s in calib test; do run $D/smartturn/$s --fast --probe --out $D/smartturn/$s/probe.jsonl; done
for s in open mcq; do run $D/e4/$s --fast --probe --out $L/e4_$s.jsonl; done

# latency, paced in real time
run $D/pack/qa --max_tokens 64 --out $L/e5_lcp.jsonl
run $D/pack/qa --max_tokens 64 --policy la2 --step 0.5 --out $L/e5_la2.jsonl
run $D/e4/open --limit 40 --endpoint --max_tokens 16 --out $L/e5_live.jsonl
# the same Listener with passes only at silence onset: a cascade that starts in the trailing silence
run $D/pack/qa --step 1000 --window 1000 --max_tokens 64 --out $L/e5_onset.jsonl
run $D/e4/open --limit 40 --endpoint --step 1000 --window 1000 --max_tokens 16 --out $L/e5_onset_live.jsonl
run $D/e4/mcq --endpoint --max_tokens 16 --out $L/e9_mcq_live.jsonl
run $D/e4/mcq --endpoint --step 1000 --window 1000 --max_tokens 16 --out $L/e9_mcq_onset_live.jsonl
cascade $D/pack/qa --max_tokens 64 --out $L/e5_parakeet.jsonl
[ -z "${WPY:-}" ] || $WPY bench/cascade.py $D/pack/qa --asr mlx-community/whisper-tiny --max_tokens 64 --out $L/e5_whisper_tiny.jsonl
for M in $Q4 $LL; do run $D/pack/qa --max_tokens 64 --llm $M --out $L/e7_lat_${M#*/}.jsonl; done

# accuracy (oracle end of speech): stream, offline cascade, written prompt; other LLMs; BBH and IFEval
run $D/vb_dev/qa --fast --out $L/e6_stream.jsonl
cascade $D/vb_dev/qa --out $L/e6_cascade.jsonl
cascade $D/vb_dev/qa --text $D/vb_dev/refs.jsonl --out $L/e6_text.jsonl
cascade $D/vb_dev/qa --asr mlx-community/parakeet-tdt-0.6b-v3 --out $L/e6_cascade_06b.jsonl
$PY -c "import json; [print(json.dumps({'file': r['file'], 'prompt': r['transcript']}))
  for r in map(json.loads, open('$L/e6_cascade.jsonl'))]" > $L/e6_cascade_prompts.jsonl
for M in $Q4 $LL; do n=${M#*/}
  run $D/vb300 --fast --llm $M --out $L/e7_acc_$n.jsonl
  cascade $D/vb300 --text $L/e6_cascade_prompts.jsonl --llm $M --out $L/e7_cascade_$n.jsonl
  cascade $D/vb300 --text $D/vb_dev/refs.jsonl --llm $M --out $L/e7_text_$n.jsonl
done
for f in flat spoken; do $PY bench/prompts.py $f $D/vb_dev/refs.jsonl > $L/${f}_prompts.jsonl
  cascade $D/vb300 --text $L/${f}_prompts.jsonl --llm $Q4 --out $L/e7_${f}_${Q4#*/}.jsonl; done
for T in bbh ifeval; do
  cascade $D/$T/qa --text $D/$T/refs.jsonl --out $L/e8_${T}_text.jsonl
  run $D/$T/qa --fast --out $L/e8_${T}_stream.jsonl
  cascade $D/$T/qa --out $L/e8_${T}_cascade.jsonl
done

# checks quoted in the text
$PY bench/checks.py qend | tee $L/checks_qend.log
$PY bench/checks.py chunking $L/e5_lcp.jsonl --n 20 | tee $L/checks_chunking.log

# the tables
A="$PY bench/analyze.py"
$A stability $L/e1_lcp.jsonl $L/e1_lcp_0.5.jsonl $L/e1_lcp_fp32.jsonl
$PY -c "import json, sys; refs = {json.loads(l)['file'] for l in open('$D/vb_dev/refs.jsonl')}
[sys.stdout.write(l) for l in open('$L/e1_lcp.jsonl') if json.loads(l)['file'] in refs]" > $L/e1_vb200.jsonl
$A fidelity $D/vb_dev/refs.jsonl $L/e1_vb200.jsonl $L/e2_110m_w12.jsonl $L/e2_06b_w8.jsonl
[ ! -f $L/e2_libritts.jsonl ] || $A fidelity $D/pack/refs.jsonl $L/e2_libritts.jsonl
$A completion $D/smartturn/calib $D/smartturn/test
[ -z "${STPY:-}" ] || $STPY bench/smartturn_ref.py $D/smartturn/test
$A endpoint $D/e4/open/labels.jsonl $L/e4_open.jsonl
$A endpoint $D/e4/mcq/labels.jsonl $L/e4_mcq.jsonl
$A latency $L/e5_*.jsonl $L/e7_lat_*.jsonl $L/e9_*.jsonl
$A accuracy $D/vb_dev/refs.jsonl $L/e6_cascade.jsonl $L/e6_stream.jsonl $L/e6_text.jsonl $L/e6_cascade_06b.jsonl
for M in $Q4 $LL; do n=${M#*/}
  $A accuracy $D/vb_dev/refs.jsonl $L/e7_cascade_$n.jsonl $L/e7_acc_$n.jsonl $L/e7_text_$n.jsonl; done
$A accuracy $D/vb_dev/refs.jsonl $L/e7_text_${Q4#*/}.jsonl $L/e7_flat_${Q4#*/}.jsonl $L/e7_spoken_${Q4#*/}.jsonl
$A numbers $D/vb_dev/refs.jsonl $L/e7_flat_${Q4#*/}.jsonl $L/e7_spoken_${Q4#*/}.jsonl
if [ -n "${VPY:-}" ]; then
  VOICEBENCH=$VOICEBENCH $VPY bench/vb_score.py bbh $D/bbh/refs.jsonl $L/e8_bbh_{cascade,stream,text}.jsonl --tasks
  VOICEBENCH=$VOICEBENCH $VPY bench/vb_score.py ifeval $D/ifeval/refs.jsonl $L/e8_ifeval_{cascade,stream,text}.jsonl
fi
