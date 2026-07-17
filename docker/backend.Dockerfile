ARG PYTORCH_IMAGE=dustynv/pytorch:2.1-r35.4.1
FROM ${PYTORCH_IMAGE}

WORKDIR /app

# 8.7 = Jetson Orin (AGX/NX/Nano) compute capability
ARG TORCH_ARCH="8.7"
ARG GROUNDED_SAM2_REPO=https://github.com/Baijing0817/Grounded-SAM-2.git
ARG GROUNDED_SAM2_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
    FORCE_CUDA=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cuda

# Mirrors the Grounded-SAM-2 Dockerfile approach: CUDA devel image plus compiler,
# OpenCV runtime libraries, ffmpeg, git, ninja, and setuptools pinned below 75.9.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    ffmpeg \
    g++ \
    gcc \
    git \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    libsm6 \
    libxext6 \
    lsof \
    ninja-build \
    patch \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# dustynv/pytorch base images don't reliably alias `python` to the python3
# that has pip/torch installed; force it so the rest of this file can keep
# using plain `python`.
RUN ln -sf "$(command -v python3)" /usr/bin/python

RUN python -m pip install --upgrade pip "setuptools>=62.3.0,<75.9" wheel

# Install the official Grounded-SAM-2 environment first, then layer AutoYOLO.
RUN git clone --depth 1 --branch ${GROUNDED_SAM2_REF} ${GROUNDED_SAM2_REPO} /opt/Grounded-SAM-2 \
    && cd /opt/Grounded-SAM-2 \
    && if [ -f requirements.txt ]; then python -m pip install --no-cache-dir -r requirements.txt; fi \
    && if [ -f grounding_dino/requirements.txt ]; then python -m pip install --no-cache-dir -r grounding_dino/requirements.txt; fi \
    && python -m pip install --no-cache-dir --no-build-isolation -e . \
    && if [ -d grounding_dino ]; then python -m pip install --no-cache-dir --no-build-isolation -e grounding_dino; fi

COPY backend/requirements.txt backend/requirements-grounded-sam.txt ./
RUN grep -vE '^(torch|torchvision|torchaudio|sam2)([=>= ]|$)' requirements.txt > /tmp/requirements-app.txt \
    && grep -vE '^(torch|torchvision|torchaudio|sam2)([=>= ]|$)' requirements-grounded-sam.txt > /tmp/requirements-grounded-sam-app.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-app.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-grounded-sam-app.txt

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/logs /app/uploads /app/model /app/training_runs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache

COPY backend/decord_stub.py /tmp/decord.py
RUN python -c "import shutil, site; shutil.copy('/tmp/decord.py', site.getsitepackages()[0] + '/decord.py')"
COPY --chown=appuser:appuser backend/app ./app
COPY --chown=appuser:appuser backend/grounded_sam2_server.py .
COPY --chown=appuser:appuser backend/alembic.ini .
COPY --chown=appuser:appuser backend/alembic ./alembic

USER appuser

ARG GROUNDING_MODEL_ID=IDEA-Research/grounding-dino-tiny
ARG SAM2_MODEL_ID=facebook/sam2.1-hiera-base-plus
ARG PRELOAD_MODELS=1

ENV GROUNDED_SAM2_GROUNDING_MODEL_ID=${GROUNDING_MODEL_ID} \
    GROUNDED_SAM2_SAM2_MODEL_ID=${SAM2_MODEL_ID}

RUN if [ "${PRELOAD_MODELS}" = "1" ]; then \
        python -c "import os; from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; from sam2.build_sam import build_sam2_hf; grounding_model_id = os.environ['GROUNDED_SAM2_GROUNDING_MODEL_ID']; sam2_model_id = os.environ['GROUNDED_SAM2_SAM2_MODEL_ID']; print(f'Preloading GroundingDINO model: {grounding_model_id}'); AutoProcessor.from_pretrained(grounding_model_id); AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id); print(f'Preloading SAM2 model: {sam2_model_id}'); build_sam2_hf(sam2_model_id, device='cpu'); print('Grounded-SAM model preload complete')" ; \
    fi

EXPOSE 8000 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
