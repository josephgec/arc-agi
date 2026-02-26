#!/usr/bin/env bash
# start_ollama.sh — Launch Ollama with settings optimised for Mac Studio 48 GB
#
# OLLAMA_NUM_PARALLEL=3  keeps all three role models (Hypothesizer, Coder,
#   Critic) loaded concurrently so there is no cold-load delay mid-solve.
#
# OLLAMA_MAX_VRAM pins the memory ceiling to 48 GB of unified RAM, preventing
#   Ollama from spilling to swap when multiple large models are resident.
#
# Usage:
#   source start_ollama.sh     # exports vars into the current shell, then starts
#   ./start_ollama.sh          # starts Ollama in a subshell (vars not exported)

export OLLAMA_NUM_PARALLEL=3
export OLLAMA_MAX_VRAM=51539607552   # 48 GiB expressed in bytes

echo "Ollama env:"
echo "  OLLAMA_NUM_PARALLEL = $OLLAMA_NUM_PARALLEL"
printf "  OLLAMA_MAX_VRAM     = %s (%.0f GiB)\n" \
    "$OLLAMA_MAX_VRAM" "$(echo "scale=1; $OLLAMA_MAX_VRAM / 1073741824" | bc)"

# Pull the three default models if not already present
MODELS=(
    "deepseek-r1:32b"
    "qwen2.5-coder:14b"
)

echo ""
echo "Checking models..."
for m in "${MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "^${m}"; then
        echo "  [ok] $m"
    else
        echo "  [pulling] $m ..."
        ollama pull "$m"
    fi
done

echo ""
echo "Starting Ollama server..."
ollama serve
