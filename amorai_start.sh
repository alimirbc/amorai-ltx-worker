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

echo "[AmorAI] Updating ComfyUI and custom nodes to latest..."
cd /comfyui && git pull origin master --ff-only 2>&1 || echo "[AmorAI] ComfyUI git pull failed, using cached version"
cd /comfyui/custom_nodes/ComfyUI-KJNodes && git pull origin main --ff-only 2>&1 || echo "[AmorAI] KJNodes git pull failed, using cached version"
echo "[AmorAI] ComfyUI-LTXVideo pinned to Docker image version (no runtime update)"

echo "[AmorAI] Launching RunPod handler..."
exec python3 -u /handler.py
