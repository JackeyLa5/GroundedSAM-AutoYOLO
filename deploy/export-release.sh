#!/usr/bin/env bash
# Build the embedded-model images and create a self-contained release folder.
# It never deletes an existing output directory.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
release_dir="${1:-${project_root}/dist/autoyolo-jetson-offline-$(date +%Y%m%d-%H%M%S)}"

sam_checkpoint="${script_dir}/models/sam2/sam2.1_hiera_small.pt"
dino_dir="${script_dir}/models/grounding-dino-tiny"

if [ ! -s "${sam_checkpoint}" ] \
    || [ ! -f "${dino_dir}/config.json" ] \
    || { [ ! -f "${dino_dir}/model.safetensors" ] && [ ! -f "${dino_dir}/pytorch_model.bin" ]; }; then
    echo "Model resources are incomplete. Run ./deploy/prepare-models.sh first." >&2
    exit 1
fi

if [ -e "${release_dir}" ]; then
    echo "Refusing to overwrite existing release directory: ${release_dir}" >&2
    exit 1
fi

cd "${project_root}"
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.build.yml build backend frontend

mkdir -p "${release_dir}/images"
docker save -o "${release_dir}/images/autoyolo-jetson-images.tar" \
    autoyolo-backend:jetson-arm64-cuda-offline \
    autoyolo-frontend:jetson-arm64-offline \
    postgres:16-alpine
(cd "${release_dir}/images" && sha256sum autoyolo-jetson-images.tar > SHA256SUMS)

cp "${script_dir}/docker-compose.yml" "${release_dir}/docker-compose.yml"
cp "${script_dir}/README_ZH.md" "${release_dir}/README_ZH.md"
cp "${script_dir}/install-on-orin.sh" "${release_dir}/install-on-orin.sh"
chmod +x "${release_dir}/install-on-orin.sh"

echo
echo "Release created: ${release_dir}"
echo "Copy this whole directory to a compatible new Orin, then run:"
echo "  cd $(basename "${release_dir}")"
echo "  ./install-on-orin.sh"
