# The Jetson PyTorch image has CUDA-enabled torch 2.1.  Install the matching
# aarch64 torchvision wheel below; the r35.4.1 torchvision image itself only
# ships torch 2.0 / torchvision 0.15.1.
ARG PYTORCH_IMAGE=dustynv/pytorch:2.1-r35.4.1
FROM ${PYTORCH_IMAGE}

WORKDIR /app

# 8.7 = Jetson Orin (AGX/NX/Nano) compute capability
ARG TORCH_ARCH="8.7"
ARG GROUNDED_SAM2_REPO=https://github.com/Baijing0817/Grounded-SAM-2.git
ARG GROUNDED_SAM2_REF=main
ARG TORCH_VERSION=2.1.0a0+41361538.nv23.06
ARG TORCHVISION_VERSION=0.16.2+c6f3977
ARG TORCHVISION_WHEEL=https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.16.2%2Bc6f3977-cp38-cp38-linux_aarch64.whl

# Keep the Grounded-SAM-2 source importable for the non-root runtime user.
# Its editable installation points to this directory rather than copying the
# `sam2` package into site-packages.
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
    FORCE_CUDA=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PYTHONPATH=/app:/opt/Grounded-SAM-2 \
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

# Do not let PyPI resolve a CPU-only torch/torchvision wheel for aarch64.
# The constraints are passed to every later pip invocation in this image.
RUN python -m pip install --no-cache-dir --no-deps "${TORCHVISION_WHEEL}" \
    && printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /tmp/jetson-torch-constraints.txt

# Install the official Grounded-SAM-2 environment first, then layer AutoYOLO.
RUN git clone --depth 1 --branch ${GROUNDED_SAM2_REF} ${GROUNDED_SAM2_REPO} /opt/Grounded-SAM-2 \
    && cd /opt/Grounded-SAM-2 \
    && if [ -f requirements.txt ]; then python -m pip install --no-cache-dir -c /tmp/jetson-torch-constraints.txt -r requirements.txt; fi \
    && if [ -f grounding_dino/requirements.txt ]; then python -m pip install --no-cache-dir -c /tmp/jetson-torch-constraints.txt -r grounding_dino/requirements.txt; fi \
    && python -m pip install --no-cache-dir --no-build-isolation -c /tmp/jetson-torch-constraints.txt -e . \
    && if [ -d grounding_dino ]; then python -m pip install --no-cache-dir --no-build-isolation -c /tmp/jetson-torch-constraints.txt -e grounding_dino; fi \
    && chmod -R a+rX /opt/Grounded-SAM-2 \
    && PYTHONPATH=/opt/Grounded-SAM-2 python -c "import sam2; print(sam2.__file__)"

COPY backend/requirements.txt backend/requirements-grounded-sam.txt ./
RUN grep -vE '^(torch|torchvision|torchaudio|sam2)([=>= ]|$)' requirements.txt > /tmp/requirements-app.txt \
    && grep -vE '^(torch|torchvision|torchaudio|sam2)([=>= ]|$)' requirements-grounded-sam.txt > /tmp/requirements-grounded-sam-app.txt \
    && python -m pip install --no-cache-dir -c /tmp/jetson-torch-constraints.txt -r /tmp/requirements-app.txt \
    && python -m pip install --no-cache-dir -c /tmp/jetson-torch-constraints.txt -r /tmp/requirements-grounded-sam-app.txt \
    && python -c "import torch, torchvision; assert torch.version.cuda and torch.backends.cuda.is_built(), f'CPU-only PyTorch was installed: {torch.__version__}'; assert torch.__version__.startswith('2.1') and torchvision.__version__.startswith('0.16.2'), f'Unexpected Jetson torch stack: torch={torch.__version__}, torchvision={torchvision.__version__}'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, torchvision {torchvision.__version__}')"

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/logs /app/uploads /app/model /app/training_runs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache

COPY backend/decord_stub.py /tmp/decord.py
RUN python -c "import shutil, site; shutil.copy('/tmp/decord.py', site.getsitepackages()[0] + '/decord.py')"
COPY --chown=appuser:appuser backend/app ./app
COPY --chown=appuser:appuser backend/grounded_sam2_server.py .
COPY --chown=appuser:appuser backend/alembic.ini .
COPY --chown=appuser:appuser backend/alembic ./alembic

# A release build can set BUNDLE_MODELS=1 and put the two verified model
# resources below deploy/models/.  The directory is always copied so the
# normal development image remains buildable even when it only contains the
# tracked .gitkeep placeholders.  Actual weight files are intentionally
# ignored by Git.
COPY --chown=appuser:appuser deploy/models/ /opt/models/

USER appuser

ARG GROUNDING_MODEL_ID=IDEA-Research/grounding-dino-tiny
ARG SAM2_MODEL_ID=facebook/sam2.1-hiera-small
ARG PRELOAD_MODELS=1
ARG BUNDLE_MODELS=0

ENV GROUNDED_SAM2_BUNDLED_GROUNDING_MODEL_PATH=/opt/models/grounding-dino-tiny \
    GROUNDED_SAM2_BUNDLED_SAM2_CHECKPOINT_PATH=/opt/models/sam2/sam2.1_hiera_small.pt

ENV GROUNDED_SAM2_GROUNDING_MODEL_ID=${GROUNDING_MODEL_ID} \
    GROUNDED_SAM2_SAM2_MODEL_ID=${SAM2_MODEL_ID}

RUN if [ "${BUNDLE_MODELS}" = "1" ]; then \
        test -s "${GROUNDED_SAM2_BUNDLED_SAM2_CHECKPOINT_PATH}"; \
        test -f "${GROUNDED_SAM2_BUNDLED_GROUNDING_MODEL_PATH}/config.json"; \
        test -f "${GROUNDED_SAM2_BUNDLED_GROUNDING_MODEL_PATH}/preprocessor_config.json"; \
        test -f "${GROUNDED_SAM2_BUNDLED_GROUNDING_MODEL_PATH}/model.safetensors" \
          -o -f "${GROUNDED_SAM2_BUNDLED_GROUNDING_MODEL_PATH}/pytorch_model.bin"; \
    fi \
    && if [ "${PRELOAD_MODELS}" = "1" ]; then \
        python -c "import os; from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; from sam2.build_sam import build_sam2_hf; grounding_model_id = os.environ['GROUNDED_SAM2_GROUNDING_MODEL_ID']; sam2_model_id = os.environ['GROUNDED_SAM2_SAM2_MODEL_ID']; print(f'Preloading GroundingDINO model: {grounding_model_id}'); AutoProcessor.from_pretrained(grounding_model_id); AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id); print(f'Preloading SAM2 model: {sam2_model_id}'); build_sam2_hf(sam2_model_id, device='cpu'); print('Grounded-SAM model preload complete')" ; \
    fi

EXPOSE 8000 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
