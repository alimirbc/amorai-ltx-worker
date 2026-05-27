#!/bin/bash
# AmorAI LTX Worker — Model Download Script
# Models download once to the network volume, then persist across all serverless workers.

set -e

download() {
    local url="$1"
    local out="$2"
    if [ ! -f "$out" ]; then
        echo "  ↳ Downloading $(basename $out)..."
        mkdir -p "$(dirname $out)"
        if command -v aria2c &>/dev/null; then
            aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
                -d "$(dirname $out)" -o "$(basename $out)" "$url"
        else
            wget -q --show-progress -O "$out" "$url" || curl -L -o "$out" "$url"
        fi
    else
        echo "  ✓ $(basename $out) already present"
    fi
}

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

SULPHUR_BASE="https://huggingface.co/SulphurAI/Sulphur-2-base/resolve/main"
TENEROS_BASE="https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main"
LIGHTRICKS_BASE="https://huggingface.co/Lightricks/LTX-2.3/resolve/main"
KIJAI_BASE="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"

###############################################################################
# Shared: Text Encoders
###############################################################################
echo "[INFO] === Text Encoders ==="
download "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
         "$TXT/gemma_3_12B_it_fp4_mixed.safetensors"

download "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors" \
         "$TXT/gemma_3_12B_it_fp8_e4m3fn.safetensors"

###############################################################################
# Shared: LoRAs
###############################################################################
echo "[INFO] === Shared LoRAs ==="
download "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
         "$LORA/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"

download "${LIGHTRICKS_BASE}/ltx-2.3-22b-distilled-lora-384.safetensors" \
         "$LORA/ltx-2.3-22b-distilled-lora-384.safetensors"

###############################################################################
# Shared: VAE
###############################################################################
echo "[INFO] === VAE ==="
download "${KIJAI_BASE}/vae/taeltx2_3.safetensors" \
         "$VAE/taeltx2_3.safetensors"

download "${KIJAI_BASE}/vae/LTX23_video_vae_bf16.safetensors" \
         "$VAE/LTX23_video_vae_bf16.safetensors"

download "${KIJAI_BASE}/vae/LTX23_audio_vae_bf16.safetensors" \
         "$VAE/LTX23_audio_vae_bf16.safetensors"

###############################################################################
# LTX-2.3: Upscalers
###############################################################################
echo "[INFO] === Upscalers ==="
download "${LIGHTRICKS_BASE}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
         "$LATENT_UP/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

download "${LIGHTRICKS_BASE}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
         "$LATENT_UP/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

###############################################################################
# Sulphur-2 NSFW Models (core set — fp8 only to save space)
###############################################################################
echo "[INFO] === Sulphur-2 Models ==="
download "${SULPHUR_BASE}/sulphur_dev_fp8mixed.safetensors" \
         "$CKPT/sulphur_dev_fp8mixed.safetensors"

download "${SULPHUR_BASE}/sulphur_lora_rank_768.safetensors" \
         "$LORA/sulphur_lora_rank_768.safetensors"

download "${SULPHUR_BASE}/distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" \
         "$LORA_LTX23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"

###############################################################################
# Edit-Anything LoRA (required by Sulphur workflows)
###############################################################################
echo "[INFO] === Edit-Anything LoRA ==="
download "https://huggingface.co/Alissonerdx/LTX-LoRAs/resolve/main/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors" \
         "$LORA_LTX23/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors"

###############################################################################
# 10Eros NSFW Model (optional, ~29GB)
###############################################################################
if [ "${DOWNLOAD_10EROS:-true}" == "true" ]; then
    echo "[INFO] === 10Eros Model ==="
    download "${TENEROS_BASE}/10Eros_v1-fp8mixed_learned.safetensors" \
             "$CKPT/10Eros_v1-fp8mixed_learned.safetensors"
fi

###############################################################################
# LTX-2.3 Full Dev FP8 (optional, ~29GB)
###############################################################################
if [ "${DOWNLOAD_LTX23_FULL_FP8:-false}" == "true" ]; then
    echo "[INFO] === LTX-2.3 Full Dev FP8 ==="
    download "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
             "$CKPT/ltx-2.3-22b-dev-fp8.safetensors"
fi

###############################################################################
# Sulphur Prompt Enhancer (optional, ~10GB)
###############################################################################
if [ "${DOWNLOAD_PROMPT_ENHANCER:-true}" == "true" ]; then
    echo "[INFO] === Sulphur Prompt Enhancer ==="
    download "${SULPHUR_BASE}/prompt_enhancer/sulphur_prompt_enhancer_model-q8_0.gguf" \
             "$PE/sulphur_prompt_enhancer_model-q8_0.gguf"
    download "${SULPHUR_BASE}/prompt_enhancer/mmproj-BF16.gguf" \
             "$PE/mmproj-BF16.gguf"
fi

echo "[INFO] All model downloads complete."
