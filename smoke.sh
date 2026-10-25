
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== Building wheel ==="
mkdir -p dist
rm -f dist/*.whl

/tmp/build-venv/bin/pip wheel . \
    --no-deps \
    --no-build-isolation \
    -w dist

echo "=== Installing wheel ==="
/tmp/try-venv/bin/pip install \
    --force-reinstall \
    --no-deps \
    dist/*.whl

echo "=== Running tests ==="
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT

cp -r tests "$TEST_DIR/tests"
cd "$TEST_DIR"

/tmp/try-venv/bin/python -m pytest -v tests

echo "=== All tests passed ==="
