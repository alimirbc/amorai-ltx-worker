FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COMFYUI_PATH=/comfyui

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    git wget curl aria2 \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    cmake build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3

RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt

RUN pip install --no-cache-dir runpod requests

RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python

COPY handler.py /handler.py
COPY download_models.sh /download_models.sh
COPY amorai_start.sh /amorai_start.sh
RUN chmod +x /download_models.sh /amorai_start.sh

CMD ["/amorai_start.sh"]
