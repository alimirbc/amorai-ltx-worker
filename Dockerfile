FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COMFYUI_PATH=/comfyui

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl aria2 \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    cmake gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && git checkout 2acf7af8991f && cd .. && \
    pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt

RUN pip install --no-cache-dir runpod requests

RUN pip install --no-cache-dir "kornia==0.7.4" && \
    KORNIA_DIR=$(python3 -c "import kornia, pathlib; print(pathlib.Path(kornia.__file__).parent)") && \
    find "$KORNIA_DIR" -name "*.py" -exec sed -i '/^@torch\.jit\.script$/d' {} \;

RUN CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX512=OFF" pip install --no-cache-dir \
    --timeout 600 --retries 3 \
    "llama-cpp-python>=0.3.2"

COPY handler.py /handler.py
COPY download_models.sh /download_models.sh
COPY amorai_start.sh /amorai_start.sh
RUN chmod +x /download_models.sh /amorai_start.sh

CMD ["/amorai_start.sh"]
