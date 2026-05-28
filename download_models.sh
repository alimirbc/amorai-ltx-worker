#!/bin/bash
# AmorAI LTX Worker — Model Download Script
# Models download once to the network volume, then persist across all serverless workers.

set -e

VOLUME=/runpod-volume/models
CKPT=$VOLUME/checkpoints
LORA=$VOLUME/loras
LORA_LTX23=$VOLUME/loras/ltx23
VAE=$VOLUME/vae
TXT=$VOLUME/text_encoders
LATENT_UP=$VOLUME/latent_upscale_models
UPSCALE=$VOLUME/upscale_models
PE=$VOLUME/prompt_enhancer

mkdir -p "$CKPT" "$LORA" "$LORA_LTX23" "$VAE" "$TXT" "$LATENT_UP" "$UPSCALE" "$PE"

# Verify a .safetensors file has a valid header (not truncated/corrupt).
# Returns 0 if valid, 1 if corrupt or missing.
verify_safetensors() {
    local f="$1"
    [ -f "$f" ] || return 1
    python3 - "$f" <<'PYEOF'
import struct, json, sys
path = sys.argv[1]
try:
    with open(path, 'rb') as h:
        raw = h.read(8)
        if len(raw) < 8:
            sys.exit(1)
        n = struct.unpack('<Q', raw)[0]
        if n == 0 or n > 100 * 1024 * 1024:
            sys.exit(1)
        header_bytes = h.read(n)
        if len(header_bytes) < n:
            sys.exit(1)
        json.loads(header_bytes)
    sys.exit(0)
except Exception as e:
    print(f"[verify] {path}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

download() {
    local url="$1"
    local out="$2"
    local basename
    basename="$(basename "$out")"

    # For .safetensors files: verify header integrity, delete if corrupt.
    if [[ "$out" == *.safetensors ]] && [ -f "$out" ]; then
        if verify_safetensors "$out"; then
            echo "  ✓ $basename already present and valid"
            return 0
        else
            echo "  ⚠ $basename is corrupt or truncated — re-downloading..."
            rm -f "$out"
        fi
    elif [ -f "$out" ]; then
        echo "  ✓ $basename already present"
        return 0
    fi

    echo "  ↳ Downloading $basename..."
    mkdir -p "$(dirname "$out")"
    if command -v aria2c &>/dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
            -d "$(dirname "$out")" -o "$basename" "$url"
    else
        wget -q --show-progress -O "$out" "$url" || curl -L -o "$out" "$url"
    fi

    # Post-download integrity check for safetensors.
    if [[ "$out" == *.safetensors ]]; then
        if ! verify_safetensors "$out"; then
            echo "  ✗ $basename failed integrity check after download — deleting."
            rm -f "$out"
            echo "  ✗ FATAL: Could not download a valid $basename. Aborting." >&2
            exit 1
        fi
        echo "  ✓ $basename integrity verified."
    fi
}

SULPHUR_BASE="https://huggingface.co/SulphurAI/Sulphur-2-base/resolve/main"
TENEROS_BASE="https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main"
LIGHTRICKS_BASE="https://huggingface.co/Lightricks/LTX-2.3/resolve/main"
KIJAI_BASE="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"

###############################################################################
# REQUIRED: Text Encoder (fp8 — used by workflow for prompt encoding, ~12GB)
###############################################################################
echo "[INFO] === Text Encoders ==="
download "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors" \
         "$TXT/gemma_3_12B_it_fp8_e4m3fn.safetensors"

# Optional: fp4 variant (~6GB). Lower quality than fp8. Set true if you want
# to experiment with a faster/smaller text encoder. Not used by default workflow.
if [ "${DOWNLOAD_GEMMA_FP4:-false}" == "true" ]; then
    download "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
             "$TXT/gemma_3_12B_it_fp4_mixed.safetensors"
fi

###############################################################################
# REQUIRED: LoRAs
###############################################################################
echo "[INFO] === Shared LoRAs ==="
download "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
         "$LORA/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"

# Optional: older distilled LoRA (384-step). Not used — workflow uses 1.1 version.
if [ "${DOWNLOAD_DISTILLED_LORA_384:-false}" == "true" ]; then
    download "${LIGHTRICKS_BASE}/ltx-2.3-22b-distilled-lora-384.safetensors" \
             "$LORA/ltx-2.3-22b-distilled-lora-384.safetensors"
fi

###############################################################################
# REQUIRED: VAE
###############################################################################
echo "[INFO] === VAE ==="
download "${KIJAI_BASE}/vae/taeltx2_3.safetensors" \
         "$VAE/taeltx2_3.safetensors"

download "${KIJAI_BASE}/vae/LTX23_video_vae_bf16.safetensors" \
         "$VAE/LTX23_video_vae_bf16.safetensors"

download "${KIJAI_BASE}/vae/LTX23_audio_vae_bf16.safetensors" \
         "$VAE/LTX23_audio_vae_bf16.safetensors"

###############################################################################
# REQUIRED: Spatial Upscaler v1.0 (used by workflow)
# Optional: v1.1 — newer version, set true to test if quality is better
###############################################################################
echo "[INFO] === Upscalers ==="
download "${LIGHTRICKS_BASE}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
         "$LATENT_UP/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

if [ "${DOWNLOAD_UPSCALER_V11:-false}" == "true" ]; then
    download "${LIGHTRICKS_BASE}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
             "$LATENT_UP/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
fi

###############################################################################
# REQUIRED: Sulphur-2 core models
###############################################################################
echo "[INFO] === Sulphur-2 Models ==="
download "${SULPHUR_BASE}/sulphur_dev_fp8mixed.safetensors" \
         "$CKPT/sulphur_dev_fp8mixed.safetensors"

download "${SULPHUR_BASE}/sulphur_lora_rank_768.safetensors" \
         "$LORA/sulphur_lora_rank_768.safetensors"

download "${SULPHUR_BASE}/distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" \
         "$LORA_LTX23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"

###############################################################################
# REQUIRED: Edit-Anything LoRA (used by Sulphur i2v workflow)
###############################################################################
echo "[INFO] === Edit-Anything LoRA ==="
download "https://huggingface.co/Alissonerdx/LTX-LoRAs/resolve/main/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors" \
         "$LORA_LTX23/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"

###############################################################################
# Optional: 10Eros NSFW Model (~29GB)
# A separate NSFW base model. Not used by Sulphur workflow.
# Set DOWNLOAD_10EROS=true on the RunPod endpoint to enable.
###############################################################################
if [ "${DOWNLOAD_10EROS:-false}" == "true" ]; then
    echo "[INFO] === 10Eros Model ==="
    download "${TENEROS_BASE}/10Eros_v1-fp8mixed_learned.safetensors" \
             "$CKPT/10Eros_v1-fp8mixed_learned.safetensors"
fi

###############################################################################
# Optional: LTX-2.3 Full Dev FP8 (~29GB)
# Vanilla (non-NSFW) base model. Only needed if running non-Sulphur workflows.
###############################################################################
if [ "${DOWNLOAD_LTX23_FULL_FP8:-false}" == "true" ]; then
    echo "[INFO] === LTX-2.3 Full Dev FP8 ==="
    download "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
             "$CKPT/ltx-2.3-22b-dev-fp8.safetensors"
fi

###############################################################################
# Optional: Sulphur Prompt Enhancer (~10GB)
# Enhances short user prompts into detailed video descriptions before generation.
# Requires llama-cpp-python. Set DOWNLOAD_PROMPT_ENHANCER=false to skip.
###############################################################################
if [ "${DOWNLOAD_PROMPT_ENHANCER:-true}" == "true" ]; then
    echo "[INFO] === Sulphur Prompt Enhancer ==="
    download "${SULPHUR_BASE}/prompt_enhancer/sulphur_prompt_enhancer_model-q8_0.gguf" \
             "$PE/sulphur_prompt_enhancer_model-q8_0.gguf"
    download "${SULPHUR_BASE}/prompt_enhancer/mmproj-BF16.gguf" \
             "$PE/mmproj-BF16.gguf"
fi

echo "[INFO] All model downloads complete."
