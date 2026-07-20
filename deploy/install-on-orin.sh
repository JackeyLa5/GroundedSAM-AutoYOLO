#!/usr/bin/env bash
# Run on a new Jetson Orin after copying the generated release directory.
# This validates host prerequisites, imports the offline images, and starts it.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive="${script_dir}/images/autoyolo-jetson-images.tar"
checksum_file="${script_dir}/images/SHA256SUMS"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "This release is for Jetson arm64 (aarch64), not: $(uname -m)" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed on this Orin. Install Docker Engine first." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose V2 is unavailable. Install docker-compose-plugin first." >&2
    exit 1
fi

if [ ! -e /dev/nvmap ]; then
    echo "/dev/nvmap is missing. Check that JetPack/NVIDIA drivers are installed." >&2
    exit 1
fi

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia'; then
    echo "NVIDIA Container Runtime is not registered with Docker." >&2
    echo "Install/configure it before starting this GPU application." >&2
    exit 1
fi

if [ ! -f "${archive}" ]; then
    echo "Image archive not found: ${archive}" >&2
    exit 1
fi

if [ -f "${checksum_file}" ]; then
    (cd "${script_dir}/images" && sha256sum -c SHA256SUMS)
fi

if [ ! -e "${script_dir}/.env" ]; then
    printf 'JETSON_GPU_GID=%s\n' "$(stat -c '%g' /dev/nvmap)" > "${script_dir}/.env"
fi

cd "${script_dir}"
docker load -i "${archive}"
docker compose up -d
docker compose ps

echo
echo "Started. Verify CUDA with:"
echo "  docker compose exec backend python -c 'import torch; print(torch.version.cuda); print(torch.cuda.is_available())'"
