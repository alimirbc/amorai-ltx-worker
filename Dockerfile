FROM runpod/worker-comfyui:latest

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt && \
    git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git || true

COPY download_models.sh /download_models.sh
COPY amorai_start.sh /amorai_start.sh
RUN chmod +x /download_models.sh /amorai_start.sh

CMD ["/amorai_start.sh"]
