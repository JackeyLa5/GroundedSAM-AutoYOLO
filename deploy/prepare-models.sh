#!/usr/bin/env bash
# Stage the two inference models in deploy/models for an embedded-model image.
# Run this on the already working, internet-capable Jetson Orin.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
models_dir="${script_dir}/models"
sam_source="${1:-/home/galbot/sam2/checkpoints/sam2.1_hiera_small.pt}"
backend_image="${BACKEND_IMAGE:-autoyolo-backend:jetson-arm64-cuda}"

if [ ! -s "${sam_source}" ]; then
    echo "SAM2 checkpoint was not found or is empty: ${sam_source}" >&2
    echo "Pass its path as the first argument, for example:" >&2
    echo "  $0 /path/to/sam2.1_hiera_small.pt" >&2
    exit 1
fi

mkdir -p "${models_dir}/sam2" "${models_dir}/grounding-dino-tiny"
cp "${sam_source}" "${models_dir}/sam2/sam2.1_hiera_small.pt"
echo "SAM2 checkpoint staged."

if [ -f "${models_dir}/grounding-dino-tiny/config.json" ] \
    && { [ -f "${models_dir}/grounding-dino-tiny/model.safetensors" ] \
         || [ -f "${models_dir}/grounding-dino-tiny/pytorch_model.bin" ]; }; then
    echo "GroundingDINO directory already exists; keeping it unchanged."
else
    echo "Downloading the complete GroundingDINO repository into deploy/models/..."
    echo "This requires internet access only on the release-building machine."
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        --env HF_HOME=/tmp/huggingface \
        --volume "${models_dir}:/output" \
        "${backend_image}" \
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IDEA-Research/grounding-dino-tiny', local_dir='/output/grounding-dino-tiny')"
fi

echo
echo "Model resources are ready in: ${models_dir}"
echo "Next: cd ${project_root} && docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.build.yml build"
