#!/bin/bash
set -e

VOLUME=/workspace
COMFYUI_MODELS=/comfyui/models

echo "[AmorAI] Setting up volume model directories..."
mkdir -p $VOLUME/models/{checkpoints,vae,text_encoders,loras/ltx23,latent_upscale_models,upscale_models,diffusion_models,prompt_enhancer}

echo "[AmorAI] Symlinking ComfyUI model dirs to volume..."
for dir in checkpoints vae text_encoders loras latent_upscale_models upscale_models diffusion_models prompt_enhancer; do
    if [ ! -L "$COMFYUI_MODELS/$dir" ]; then
        rm -rf "$COMFYUI_MODELS/$dir"
        ln -sfn "$VOLUME/models/$dir" "$COMFYUI_MODELS/$dir"
        echo "  ✓ Linked $dir → $VOLUME/models/$dir"
    else
        echo "  ✓ $dir already linked"
    fi
done

echo "[AmorAI] Downloading models (skips existing files)..."
bash /download_models.sh

echo "[AmorAI] Launching RunPod ComfyUI handler..."
exec /start.sh
