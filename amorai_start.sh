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

LTXVIDEO_DIR=/comfyui/custom_nodes/ComfyUI-LTXVideo

echo "[AmorAI] Updating ComfyUI-LTXVideo to latest main..."
PREV_COMMIT=$(git -C "$LTXVIDEO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
if git -C "$LTXVIDEO_DIR" fetch --quiet origin 2>/dev/null; then
    if git -C "$LTXVIDEO_DIR" checkout -B main origin/main 2>/dev/null; then
        NEW_COMMIT=$(git -C "$LTXVIDEO_DIR" rev-parse HEAD 2>/dev/null || echo "$PREV_COMMIT")
        echo "[AmorAI]   ComfyUI-LTXVideo: ${PREV_COMMIT:0:10} → ${NEW_COMMIT:0:10}"
        if [ "$PREV_COMMIT" != "$NEW_COMMIT" ]; then
            echo "[AmorAI]   Installing updated requirements..."
            pip install --quiet --no-cache-dir -r "$LTXVIDEO_DIR/requirements.txt" 2>&1 | tail -3 || true
        else
            echo "[AmorAI]   Already at latest."
        fi
    else
        echo "[AmorAI]   WARNING: checkout failed, using pinned ${PREV_COMMIT:0:10}"
    fi
else
    echo "[AmorAI]   WARNING: fetch failed (network?), using pinned ${PREV_COMMIT:0:10}"
fi

echo "[AmorAI]   ComfyUI          — pinned (master at image build time)"
echo "[AmorAI]   ComfyUI-KJNodes  — pinned to 6dd3c674"

echo "[AmorAI] Launching RunPod handler..."
exec python3 -u /handler.py
