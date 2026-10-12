#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPT-6 Luna and Kimi K3 have already been evaluated. Keep the remaining
# OpenCode Go frame-QA configs explicit so this submission cannot rerun them.
export CONFIGS="configs/qa_grok_4_7.json configs/qa_qwen3_8_max.json configs/qa_deepseek_v4_flash_vision_exp.json"

printf 'Submitting remaining OpenCode Go models: %s\n' "$CONFIGS"
exec qsub "$SCRIPT_DIR/run_all_eval.sh"
