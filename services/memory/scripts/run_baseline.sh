#!/usr/bin/env bash
# ScribeMem-Bench: end-to-end baseline pipeline
#
# Stages:
#   1. Load scenes2 data (embodied mode) → MemoryService
#   2. Generate bilingual QA pairs with LLM
#   3. Evaluate: RAG + VLM-QA + LLM-Judge
#   4. Generate baseline_report.md + metrics.json
#
# Usage:
#   ./scripts/run_baseline.sh                          # default paths
#   ./scripts/run_baseline.sh --session session_002     # specific session
#   DRY_RUN=1 ./scripts/run_baseline.sh                # print commands only
#
# Env vars:
#   VLM_API_KEY              DeepSeek API key (required for QA gen + eval LLM judge)
#   VLM_BASE_URL             DeepSeek base URL (default: https://api.deepseek.com)
#   VLM_MODEL                Model name (default: deepseek-v4-flash)
#   MEM_VLM_API_KEY          Aliyun/backup VLM key (for QA images)
#   EVAL_QA_DELAY_S          Delay between QA pairs (default: 1.5)
#   MEMGRAPH_LLM_RETRIES     LLM ranker retry attempts (default: 3)
#   MEMGRAPH_BACKUP_LLM_KEY  Dedicated backup LLM ranker API key
#                             (if unset, falls back to MEM_VLM_* for Aliyun qwen)
#   MEMGRAPH_BACKUP_LLM_URL  Dedicated backup LLM ranker base URL
#                             (should differ from VLM_BASE_URL to avoid
#                              same-provider failure mode)
#   DATA_DIR                 Dataset root (default: /home/hyl/embodypaper/datasets)
#   OUTPUT_DIR               Results output dir (default: auto-dated)

set -euo pipefail

# ── Resolve paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVC_DIR="$(dirname "$SCRIPT_DIR")"
ROBONIX_ROOT="$(cd "$SVC_DIR/../.." && pwd)"

# ── Configuration ──
SESSION="${1:-session_001}"
DATA_DIR="${DATA_DIR:-/home/hyl/embodypaper/datasets}"
SESSION_DIR="${DATA_DIR}/scenes2/${SESSION}"
TIMESTAMP="$(date +%Y%m%d)"
OUTPUT_DIR="${OUTPUT_DIR:-/home/hyl/embodypaper/results/baseline_final_${TIMESTAMP}}"
MEMORY_DIR="${OUTPUT_DIR}/memory_data"
QA_DIR="${OUTPUT_DIR}/qa_output"

# LLM config (defaults)
VLM_MODEL="${VLM_MODEL:-deepseek-v4-flash}"
VLM_BASE_URL="${VLM_BASE_URL:-https://api.deepseek.com}"
MAX_QA_PER_CATEGORY="${MAX_QA_PER_CATEGORY:-5}"
REQUIRE_IMAGE_RATIO="${REQUIRE_IMAGE_RATIO:-0.5}"

# ── Validation ──
if [ ! -d "$SESSION_DIR" ]; then
    echo "ERROR: Session directory not found: $SESSION_DIR"
    echo "Available sessions:"
    ls -d "${DATA_DIR}/scenes2/"*/ 2>/dev/null || echo "  (none)"
    exit 1
fi

if [ -z "${VLM_API_KEY:-}" ]; then
    echo "ERROR: VLM_API_KEY env var is required"
    echo "Set it to your DeepSeek API key (or compatible provider)"
    exit 1
fi

# ── Print config ──
echo "=============================================="
echo "  ScribeMem-Bench Baseline Pipeline"
echo "=============================================="
echo "  Session:      $SESSION"
echo "  Session dir:  $SESSION_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Memory dir:   $MEMORY_DIR"
echo "  QA dir:       $QA_DIR"
echo "  LLM model:    $VLM_MODEL"
echo "  LLM URL:      $VLM_BASE_URL"
echo "  QA/category:  $MAX_QA_PER_CATEGORY"
echo "  Image ratio:  $REQUIRE_IMAGE_RATIO"
echo "=============================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 — exiting without running commands"
    exit 0
fi

# ── Setup ──
mkdir -p "$OUTPUT_DIR" "$QA_DIR"

# Preserve captured images across runs (disable _clean_slate)
export MEMGRAPH_KEEP_DATA=1

# ── Stage 1: Load data (embodied mode) ──
echo ""
echo "=== Stage 1/4: Load scenes2 (embodied mode) ==="

cd "$SVC_DIR" && uv run python scripts/load_scenes2.py \
    --session "$SESSION_DIR" \
    --mode embodied \
    --data-dir "$MEMORY_DIR" \
    --embodied-cooldown-frames 30

# Backup captured images to output directory
IMAGES_SRC="${SVC_DIR}/data/images"
IMAGES_BAK="${OUTPUT_DIR}/images"
if [ -d "$IMAGES_SRC" ] && [ "$(ls -A "$IMAGES_SRC" 2>/dev/null)" ]; then
    mkdir -p "$IMAGES_BAK"
    cp -r "$IMAGES_SRC"/* "$IMAGES_BAK/" 2>/dev/null || true
    IMG_COUNT=$(find "$IMAGES_BAK" -type f | wc -l)
    echo "Images backed up: $IMG_COUNT files → $IMAGES_BAK"
else
    echo "WARNING: No captured images found at $IMAGES_SRC"
fi

if [ ! -f "${MEMORY_DIR}/memory_nodes.json" ]; then
    echo "ERROR: Stage 1 failed — memory_nodes.json not created"
    exit 1
fi

NODE_COUNT=$(python3 -c "import json; print(len(json.load(open('${MEMORY_DIR}/memory_nodes.json'))))")
echo "Stage 1 done: $NODE_COUNT memory nodes"

# Verify: ≥80% nodes have video_clip_refs
VIDEO_REF_COUNT=$(python3 -c "
import json
nodes = json.load(open('${MEMORY_DIR}/memory_nodes.json'))
vrefs = sum(1 for n in nodes if n.get('video_clip_refs'))
print(vrefs)
")
VIDEO_REF_PCT=$(python3 -c "print(round($VIDEO_REF_COUNT / $NODE_COUNT * 100, 1))")
echo "  Nodes with video_clip_refs: $VIDEO_REF_COUNT/$NODE_COUNT (${VIDEO_REF_PCT}%)"

# ── Stage 2: QA pairs (use canonical fixed set, generate only if missing) ──
echo ""
echo "=== Stage 2/4: QA pairs ==="

CANONICAL_QA="${SESSION_DIR}/qa_pairs"

if [ -d "$CANONICAL_QA" ] && [ "$(ls "$CANONICAL_QA"/*.json 2>/dev/null | wc -l)" -ge 6 ]; then
    echo "Using canonical QA pairs from: $CANONICAL_QA"
    cp "$CANONICAL_QA"/*.json "$CANONICAL_QA"/*.yaml "$QA_DIR/" 2>/dev/null || true
else
    echo "Canonical QA not found — generating fresh (ONE-TIME, save to dataset)"
    cd "$SVC_DIR" && uv run python scripts/generate_qa.py \
        --input "${MEMORY_DIR}/memory_nodes.json" \
        --output-dir "$QA_DIR" \
        --llm-model "$VLM_MODEL" \
        --llm-url "$VLM_BASE_URL" \
        --llm-key "$VLM_API_KEY" \
        --max-qa-per-category "$MAX_QA_PER_CATEGORY" \
        --require-image-ratio "$REQUIRE_IMAGE_RATIO"
    # Save as canonical set for future runs
    mkdir -p "$CANONICAL_QA"
    cp "$QA_DIR"/*.json "$QA_DIR"/*.yaml "$CANONICAL_QA/" 2>/dev/null || true
    echo "Canonical QA saved to: $CANONICAL_QA"
fi

QA_TOTAL=$(python3 -c "
import json, os
total = 0
for f in os.listdir('${QA_DIR}'):
    if f.endswith('.json'):
        total += len(json.load(open(os.path.join('${QA_DIR}', f))))
print(total)
")
echo "Stage 2 done: $QA_TOTAL QA pairs (canonical)"
echo "  Categories: $(ls "$QA_DIR"/*.json 2>/dev/null | wc -l)"

# ── Stage 3: Evaluate ──
echo ""
echo "=== Stage 3/4: Evaluate QA pairs ==="

cd "$SVC_DIR" && uv run python scripts/eval_qa.py \
    --qa-dir "$QA_DIR" \
    --memory-dir "$MEMORY_DIR" \
    --llm-model "$VLM_MODEL" \
    --llm-url "$VLM_BASE_URL" \
    --llm-key "$VLM_API_KEY" \
    --vlm-qa \
    --output "${OUTPUT_DIR}/metrics.json"

if [ ! -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "ERROR: Stage 3 failed — metrics.json not created"
    exit 1
fi

# ── Quick checks ──
OVERALL_ACC=$(python3 -c "
import json
m = json.load(open('${OUTPUT_DIR}/metrics.json'))
print(m['overall']['accuracy'])
")
EXISTENCE_ACC=$(python3 -c "
import json
m = json.load(open('${OUTPUT_DIR}/metrics.json'))
print(m['by_category'].get('existence_recall', {}).get('accuracy', 0))
")
VLM_UTIL=$(python3 -c "
import json
m = json.load(open('${OUTPUT_DIR}/metrics.json'))
print(m.get('vlm_util', {}).get('vlm_utilization', 0))
")

echo "Stage 3 done: overall=${OVERALL_ACC} existence_recall=${EXISTENCE_ACC} vlm_util=${VLM_UTIL}"

# ── Stage 4: Generate report ──
echo ""
echo "=== Stage 4/4: Generate baseline report ==="

cd "$SVC_DIR" && uv run python scripts/eval_qa.py \
    --qa-dir "$QA_DIR" \
    --memory-dir "$MEMORY_DIR" \
    --output "${OUTPUT_DIR}/metrics.json" \
    --generate-report \
    --report-output "${OUTPUT_DIR}/baseline_report.md"

echo ""

# ── Verification ──
PASS=true
echo "=============================================="
echo "  Verification"
echo "=============================================="

# Check 1: existence_recall ≥ 80%
if python3 -c "exit(0 if $EXISTENCE_ACC >= 0.8 else 1)"; then
    echo "  ✓ existence_recall=${EXISTENCE_ACC} ≥ 0.8"
else
    echo "  ✗ existence_recall=${EXISTENCE_ACC} < 0.8"
    PASS=false
fi

# Check 2: VLM-Util > 0
if python3 -c "exit(0 if $VLM_UTIL > 0 else 1)"; then
    echo "  ✓ vlm_utilization=${VLM_UTIL} > 0"
else
    echo "  ✗ vlm_utilization=${VLM_UTIL} = 0 (no QA used VLM)"
    PASS=false
fi

# Check 3: ≥80% nodes have video_clip_refs
if python3 -c "exit(0 if $VIDEO_REF_PCT >= 80.0 else 1)"; then
    echo "  ✓ video_clip_refs=${VIDEO_REF_PCT}% ≥ 80%"
else
    echo "  ✗ video_clip_refs=${VIDEO_REF_PCT}% < 80%"
    PASS=false
fi

echo "=============================================="
if $PASS; then
    echo "  ALL CHECKS PASSED ✓"
else
    echo "  SOME CHECKS FAILED ✗"
fi
echo ""
echo "  Results: $OUTPUT_DIR"
echo "  Report:  ${OUTPUT_DIR}/baseline_report.md"
echo "  Metrics: ${OUTPUT_DIR}/metrics.json"
echo "=============================================="
