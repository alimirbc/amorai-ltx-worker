#!/bin/bash
# AmorAI LTX Worker — Model Download Script
# Exact 1-to-1 match with community LTX 2.3 workflow template.
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
    local min_bytes="${3:-0}"   # optional minimum file size in bytes (catches aria2c pre-alloc zeros)
    local basename
    basename="$(basename "$out")"

    if [ -f "$out" ]; then
        local actual_size
        actual_size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null || echo 0)

        # Size check first: aria2c pre-allocates full file size then fills it in.
        # A file that is smaller than expected is incomplete even if the header looks valid.
        if [ "$min_bytes" -gt 0 ] && [ "$actual_size" -lt "$min_bytes" ]; then
            echo "  ⚠ $basename is too small ($(( actual_size / 1024 / 1024 ))MiB < $(( min_bytes / 1024 / 1024 ))MiB minimum) — re-downloading..."
            rm -f "$out"
        # Header integrity check for safetensors.
        elif [[ "$out" == *.safetensors ]]; then
            if verify_safetensors "$out"; then
                echo "  ✓ $basename already present and valid"
                return 0
            else
                echo "  ⚠ $basename is corrupt or truncated — re-downloading..."
                rm -f "$out"
            fi
        else
            echo "  ✓ $basename already present"
            return 0
        fi
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
        # Also verify size if minimum was specified.
        if [ "$min_bytes" -gt 0 ]; then
            local final_size
            final_size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null || echo 0)
            if [ "$final_size" -lt "$min_bytes" ]; then
                echo "  ✗ $basename is still too small after download — deleting."
                rm -f "$out"
                echo "  ✗ FATAL: $basename did not download fully. Aborting." >&2
                exit 1
            fi
        fi
        echo "  ✓ $basename integrity verified."
    fi
}

SULPHUR_BASE="https://huggingface.co/SulphurAI/Sulphur-2-base/resolve/main"
TENEROS_BASE="https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main"
LIGHTRICKS_BASE="https://huggingface.co/Lightricks/LTX-2.3/resolve/main"
KIJAI_BASE="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"

###############################################################################
# REQUIRED: Text Encoder — gemma fp8 SCALED variant (community template uses this)
# ~13 GB. The "scaled" variant uses per-tensor scaling for higher quality than e4m3fn.
###############################################################################
echo "[INFO] === Text Encoders ==="
download "${KIJAI_BASE}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
         "$TXT/gemma_3_12B_it_fp8_scaled.safetensors" \
         11811160064   # 11 GiB minimum

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
# REQUIRED: Spatial Upscaler v1.1 (community template uses this version)
###############################################################################
echo "[INFO] === Upscalers ==="
download "${LIGHTRICKS_BASE}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
         "$LATENT_UP/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

###############################################################################
# REQUIRED: 10Eros base model (~29 GB) — primary checkpoint
###############################################################################
echo "[INFO] === 10Eros Model ==="
download "${TENEROS_BASE}/10Eros_v1-fp8mixed_learned.safetensors" \
         "$CKPT/10Eros_v1-fp8mixed_learned.safetensors" \
         27917287424   # 26 GiB minimum (actual ~29.2 GiB)

###############################################################################
# REQUIRED: SulphurEXP LoRA (~2.35 GB) — applied at strength 0.3 over 10Eros
###############################################################################
echo "[INFO] === SulphurEXP LoRA ==="
download "https://huggingface.co/maximsobolev275/LTX-SulphurExperimental-LoRA-Optimized/resolve/main/LTX_SulphurEXP_LoRA_fro99-avgrank105.safetensors" \
         "$LORA/LTX_SulphurEXP_LoRA_fro99-avgrank105.safetensors"

###############################################################################
# REQUIRED: Distilled LoRAs
# First pass:  distilled-lora-1.1 @ strength 1.0
# Final pass:  distilled-lora-384-1.1 @ strength 0.5  (community uses this for refinement)
###############################################################################
echo "[INFO] === Distilled LoRAs ==="
download "${SULPHUR_BASE}/distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" \
         "$LORA_LTX23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"

download "${LIGHTRICKS_BASE}/ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
         "$LORA_LTX23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

###############################################################################
# Optional: Sulphur Prompt Enhancer (~10 GB)
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

###############################################################################
# Optional: Legacy / experimental models (disabled by default)
###############################################################################
if [ "${DOWNLOAD_SULPHUR_LEGACY:-false}" == "true" ]; then
    echo "[INFO] === Legacy Sulphur-2 Models ==="
    download "${SULPHUR_BASE}/sulphur_dev_fp8mixed.safetensors" \
             "$CKPT/sulphur_dev_fp8mixed.safetensors" \
             27917287424
    download "${SULPHUR_BASE}/sulphur_lora_rank_768.safetensors" \
             "$LORA/sulphur_lora_rank_768.safetensors" \
             9663676416
fi

if [ "${DOWNLOAD_LTX23_FULL_FP8:-false}" == "true" ]; then
    echo "[INFO] === LTX-2.3 Full Dev FP8 ==="
    download "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
             "$CKPT/ltx-2.3-22b-dev-fp8.safetensors"
fi

echo "[INFO] All model downloads complete."
