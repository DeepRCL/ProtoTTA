#!/bin/bash
# =============================================================================
# Download Stanford Dogs Dataset
# Downloads to: ProtoPFormer/datasets/stanford_dogs/
# Files: Images.tar, Annotation.tar, lists.tar  (~800MB total)
# =============================================================================

set -e

DATASET_DIR="$(dirname "$0")/../datasets/stanford_dogs"
mkdir -p "$DATASET_DIR"

BASE_URL="http://vision.stanford.edu/aditya86/ImageNetDogs"

echo "==> Downloading Stanford Dogs dataset to: $DATASET_DIR"
echo "    (Total size: ~800 MB, this may take a while)"

cd "$DATASET_DIR"

# ---------- Images ----------
if [ -d "Images" ] && [ "$(ls -1 Images | wc -l)" -eq 120 ]; then
    echo "[SKIP] Images already present (120 breed folders found)"
else
    echo "[1/3] Downloading images.tar..."
    wget -c --progress=bar "$BASE_URL/images.tar" -O images.tar
    echo "      Extracting images.tar..."
    tar -xf images.tar
    rm images.tar
    echo "      Done."
fi

# ---------- Annotations ----------
if [ -d "Annotation" ] && [ "$(ls -1 Annotation | wc -l)" -eq 120 ]; then
    echo "[SKIP] Annotations already present"
else
    echo "[2/3] Downloading annotation.tar..."
    wget -c --progress=bar "$BASE_URL/annotation.tar" -O annotation.tar
    echo "      Extracting annotation.tar..."
    tar -xf annotation.tar
    rm annotation.tar
    echo "      Done."
fi

# ---------- Train/Test splits ----------
if [ -f "train_list.mat" ] && [ -f "test_list.mat" ]; then
    echo "[SKIP] Split files (train_list.mat, test_list.mat) already present"
else
    echo "[3/3] Downloading lists.tar (train/test split .mat files)..."
    wget -c --progress=bar "$BASE_URL/lists.tar" -O lists.tar
    echo "      Extracting lists.tar..."
    tar -xf lists.tar
    rm lists.tar
    echo "      Done."
fi

echo ""
echo "==> Verifying dataset structure..."

ERRORS=0

[ -d "Images" ]         || { echo "  [ERROR] Images/ directory missing"; ERRORS=$((ERRORS+1)); }
[ -d "Annotation" ]     || { echo "  [ERROR] Annotation/ directory missing"; ERRORS=$((ERRORS+1)); }
[ -f "train_list.mat" ] || { echo "  [ERROR] train_list.mat missing"; ERRORS=$((ERRORS+1)); }
[ -f "test_list.mat" ]  || { echo "  [ERROR] test_list.mat missing"; ERRORS=$((ERRORS+1)); }

IMG_COUNT=$(ls -1 Images 2>/dev/null | wc -l)
ANN_COUNT=$(ls -1 Annotation 2>/dev/null | wc -l)

echo "  Images breeds found:     $IMG_COUNT / 120"
echo "  Annotation breeds found: $ANN_COUNT / 120"

if [ "$IMG_COUNT" -ne 120 ]; then
    echo "  [ERROR] Expected 120 breed folders in Images/, found $IMG_COUNT"
    ERRORS=$((ERRORS+1))
fi

if [ "$ERRORS" -eq 0 ]; then
    echo ""
    echo "==> Dataset ready! Structure:"
    echo "    $DATASET_DIR/"
    echo "    ├── Images/       (120 breed subdirs)"
    echo "    ├── Annotation/   (120 breed subdirs)"
    echo "    ├── train_list.mat"
    echo "    └── test_list.mat"
    echo ""
    echo "==> Next: run  sh scripts/train_dogs_small.sh"
else
    echo ""
    echo "==> Download incomplete ($ERRORS error(s)). Re-run this script."
    exit 1
fi
