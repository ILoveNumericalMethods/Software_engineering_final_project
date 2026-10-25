#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" &&
cd "$ROOT" &&

WORK_DIR="$(mktemp -d)" &&
trap 'rm -rf "$WORK_DIR"' EXIT &&

echo "=== Building wheel ===" &&
python3 -m venv "$WORK_DIR/build-venv" &&
"$WORK_DIR/build-venv/bin/pip" install "setuptools>=64" wheel numpy &&
"$WORK_DIR/build-venv/bin/pip" install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu &&

mkdir -p dist &&
rm -f dist/*.whl &&

"$WORK_DIR/build-venv/bin/pip" wheel . \
    --no-deps \
    --no-build-isolation \
    -w dist &&

ls dist | grep linux &&

echo "=== Installing wheel ===" &&
python3 -m venv "$WORK_DIR/try-venv" &&
"$WORK_DIR/try-venv/bin/pip" install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu &&
"$WORK_DIR/try-venv/bin/pip" install pytest numpy &&
"$WORK_DIR/try-venv/bin/pip" install \
    --force-reinstall \
    --no-deps \
    dist/*.whl &&

echo "=== Running tests ===" &&
TEST_DIR="$WORK_DIR/test-run" &&
mkdir -p "$TEST_DIR" &&

cp -r tests "$TEST_DIR/tests" &&
cd "$TEST_DIR" &&

"$WORK_DIR/try-venv/bin/python" -c "import torch; from fse_ops import median_filter5x5; x = torch.arange(25, dtype=torch.float32).reshape(5, 5); assert torch.equal(median_filter5x5(x), torch.tensor([[12.0]]))" &&
"$WORK_DIR/try-venv/bin/python" -m pytest -v tests &&

echo "=== All tests passed ===" &&
echo OK
