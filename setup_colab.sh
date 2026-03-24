#!/usr/bin/env bash

set -euo pipefail

WITH_MODELS=0

for arg in "$@"; do
  case "$arg" in
    --with-models)
      WITH_MODELS=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/cv-final-project}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DRIVE_ROOT/checkpoints}"
INPUT_DIR="${INPUT_DIR:-$DRIVE_ROOT/inputs}"
RESULTS_DIR="${RESULTS_DIR:-$DRIVE_ROOT/results}"

GROUNDING_DINO_TAG="${GROUNDING_DINO_TAG:-v0.1.0-alpha2}"
SAM2_REF="${SAM2_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
GROUNDING_DINO_CKPT_URL="${GROUNDING_DINO_CKPT_URL:-https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth}"
SAM2_CKPT_URL="${SAM2_CKPT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt}"
GROUNDING_DINO_CKPT_PATH="${GROUNDING_DINO_CKPT_PATH:-$CHECKPOINT_DIR/groundingdino_swint_ogc.pth}"
SAM2_CKPT_PATH="${SAM2_CKPT_PATH:-$CHECKPOINT_DIR/sam2.1_hiera_small.pt}"

download_if_missing() {
  local url="$1"
  local output_path="$2"
  if [[ -f "$output_path" ]]; then
    echo "[setup] checkpoint already present: $output_path"
    return 0
  fi
  echo "[setup] downloading $(basename "$output_path")"
  curl -L --fail --output "$output_path" "$url"
}

echo "[setup] upgrading pip"
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel

echo "[setup] installing pinned base dependencies"
$PYTHON_BIN -m pip install -r requirements.txt

echo "[setup] creating shared Drive folders"
mkdir -p "$CHECKPOINT_DIR" "$INPUT_DIR" "$RESULTS_DIR"

if [[ "$WITH_MODELS" -eq 0 ]]; then
  echo "[setup] base environment ready"
  echo "[setup] rerun with --with-models to install GroundingDINO and SAM2"
  exit 0
fi

echo "[setup] installing pinned GPU stack for D1"
$PYTHON_BIN -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

$PYTHON_BIN -m pip install \
  transformers==4.46.3 \
  accelerate==1.0.1 \
  huggingface_hub==0.26.2 \
  supervision==0.25.1

echo "[setup] installing GroundingDINO from official repo tag ${GROUNDING_DINO_TAG}"
$PYTHON_BIN -m pip install "git+https://github.com/IDEA-Research/GroundingDINO.git@${GROUNDING_DINO_TAG}"

echo "[setup] installing SAM2 from official repo ref ${SAM2_REF}"
$PYTHON_BIN -m pip install "git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}"

download_if_missing "$GROUNDING_DINO_CKPT_URL" "$GROUNDING_DINO_CKPT_PATH"
download_if_missing "$SAM2_CKPT_URL" "$SAM2_CKPT_PATH"

cat <<EOF
[setup] model stack install complete
[setup] expected checkpoint files:
  - ${GROUNDING_DINO_CKPT_PATH}
  - ${SAM2_CKPT_PATH}
[setup] you can now run notebooks/01_smoke_test.ipynb
EOF

ls -lh "$GROUNDING_DINO_CKPT_PATH" "$SAM2_CKPT_PATH"
