#!/usr/bin/env bash
# Evaluate base + the three shared-mind students on PrefEval, implicit and explicit.
#
# implicit = the preference is never stated; the model must infer it from the
#            persona and the conversation. This is what the shared-mind training
#            teaches, so it is the primary metric.
# explicit = the preference is stated outright. Both arms should handle it, so a
#            gain here would mean general quality improved rather than inference.
#
# The contrast between the two is the whole point: an advantage concentrated in
# implicit supports "the student learned to infer hidden state"; an equal
# advantage in both supports the duller "cleaner labels helped".
#
#   GPU=3 PORT=8021 bash scripts/evaluate_prefeval_arms.sh
set -uo pipefail

GPU="${GPU:-3}"
PORT="${PORT:-8021}"
REPO="$HOME/mind2dialogue/evaluations"
TRAIN_OUT="$HOME/mind2dialogue/training/outputs"
BASE="${BASE_MODEL:-$HOME/models/Qwen2.5-7B-Instruct}"
PY="${PY:-$REPO/.venv/bin/python}"
export LD_PRELOAD=""
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"
mkdir -p "$REPO/results" "$REPO/logs"

echo "[serve] vLLM on GPU $GPU port $PORT with 3 LoRA adapters"
CUDA_VISIBLE_DEVICES="$GPU" "$HOME/mind2dialogue/training/.venv/bin/vllm" serve "$BASE" \
  --served-model-name base \
  --enable-lora --max-lora-rank 64 --max-loras 3 \
  --lora-modules "shared=$TRAIN_OUT/three_arm_shared/final" \
                 "inferred=$TRAIN_OUT/three_arm_inferred/final" \
                 "blind=$TRAIN_OUT/three_arm_blind/final" \
  --max-model-len 16384 --gpu-memory-utilization 0.55 \
  --chat-template-content-format string \
  --port "$PORT" > "$REPO/logs/vllm_three_arm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "[serve] died — see logs/vllm_three_arm.log"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null || { echo "[serve] timeout"; exit 1; }
echo "[serve] ready"

for M in base blind inferred shared; do
  for PREF in implicit explicit; do
    echo "[eval] model=$M pref=$PREF  $(date +%H:%M:%S)"
    MODEL="$M" PORT="$PORT" PY="$PY" PREF_TYPE="$PREF" \
      OUT_ROOT="$REPO/results/prefeval_${PREF}" \
      bash "$REPO/scripts/run_prefeval_gen.sh" \
      > "$REPO/logs/${M}_prefeval_${PREF}.log" 2>&1
    echo "[eval] model=$M pref=$PREF exit=$?  $(date +%H:%M:%S)"
  done
done
echo "[done] results under $REPO/results/"
