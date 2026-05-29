#!/bin/bash
set -e

VOLUME=/runpod-volume
COMFYUI_MODELS=/comfyui/models

echo "[AmorAI] Setting up volume model directories..."
mkdir -p \
    $VOLUME/models/checkpoints \
    $VOLUME/models/vae \
    $VOLUME/models/text_encoders \
    $VOLUME/models/loras \
    $VOLUME/models/loras/ltx23 \
    $VOLUME/models/latent_upscale_models \
    $VOLUME/models/upscale_models \
    $VOLUME/models/diffusion_models \
    $VOLUME/models/prompt_enhancer

echo "[AmorAI] Symlinking ComfyUI model dirs to volume..."
for dir in checkpoints vae text_encoders loras latent_upscale_models upscale_models diffusion_models prompt_enhancer; do
    if [ ! -L "$COMFYUI_MODELS/$dir" ]; then
        rm -rf "$COMFYUI_MODELS/$dir"
        ln -sfn "$VOLUME/models/$dir" "$COMFYUI_MODELS/$dir"
        echo "  ✓ Linked $dir"
    else
        echo "  ✓ $dir already linked"
    fi
done

echo "[AmorAI] Downloading models (skips existing files)..."
bash /download_models.sh

echo "[AmorAI] All custom nodes pinned to Docker image versions (no runtime updates)"
echo "[AmorAI]   ComfyUI         — pinned (master at image build time)"
echo "[AmorAI]   ComfyUI-LTXVideo — pinned to 2acf7af8991f"
echo "[AmorAI]   ComfyUI-KJNodes  — pinned to 6dd3c674"

echo "[AmorAI] Launching RunPod handler..."
exec python3 -u /handler.py
