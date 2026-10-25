# median_filter5x5

CPU C++ extension, version 0.1.0. Python function: `fse_ops.median_filter5x5`.

For each image and each valid window position, return the median (the 13th
smallest value) of the 25 values in its 5×5 neighborhood. The filter uses
stride 1 and no padding. For a leading index tuple \(b\),

$y_{b,i,j}=\text{median}\{x_{b,i+u,j+v}:0\leq u,v<5\}.$

Exact one-line PyTorch reference:

```python
x.unfold(-2, 5, 1).unfold(-2, 5, 1).flatten(-2).median(dim=-1).values
```

Input: a strided CPU `float32` tensor of shape `(..., H, W)`, with `H >= 5`
and `W >= 5`. Any number of leading dimensions is supported, including no
leading dimensions and empty batch dimensions. Non-contiguous inputs are
accepted. Exact value agreement is supported for finite inputs.

Output: a CPU `float32` tensor of shape `(..., H-4, W-4)`, with every image
filtered independently.

Autograd is not implemented.

## Build and test

Run from the project root inside the prepared course Linux sandbox
(Python 3.12 and a C++ compiler).

```bash
python3 -m venv /tmp/build-venv
/tmp/build-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu /tmp/build-venv/bin/pip wheel . --no-deps -w dist

python3 -m venv /tmp/try-venv
/tmp/try-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/try-venv/bin/pip install pytest numpy
/tmp/try-venv/bin/pip install dist/*.whl
PROJECT_DIR=$(pwd)
cp -r tests /tmp/proj-tests
cd /tmp
/tmp/try-venv/bin/python -c "import fse_ops; print(fse_ops.__file__)"
/tmp/try-venv/bin/python -m pytest -q /tmp/proj-tests
cd "$PROJECT_DIR"
```

Expected: an import from the venv's `site-packages` and `5 passed`.
The extra CPU index also supplies torch to the isolated build. Build and
runtime PyTorch versions must match. Docker, CI, and smoke pin version 2.10.0.

Example after installation:

```bash
/tmp/try-venv/bin/python -c "import torch; from fse_ops import median_filter5x5; print(median_filter5x5(torch.arange(25, dtype=torch.float32).reshape(5, 5)))"
```

Result: `tensor([[12.]])`.

## Checks

From the project root in the same sandbox:

```bash
/tmp/build-venv/bin/pip install ruff
/tmp/build-venv/bin/ruff check .
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu make wheel PYTHON=/tmp/build-venv/bin/python
make install PYTHON=/tmp/try-venv/bin/python
make test PYTHON=/tmp/try-venv/bin/python
make clean
chmod 755 tools/smoke.sh
./tools/smoke.sh
```

The smoke script creates fresh venvs, checks the Linux wheel tag, imports
and calls the installed operation outside the checkout, runs pytest, and
prints `OK`. CI runs Ruff and installed-wheel pytest on pushes and pull requests.

The five tests cover shape, dtype, exact values, mixed values/batching, and
invalid dtype. Changing `window[12]` to `window[11]` made `test_exact_values`
and `test_mixed_values_and_batch` fail. Screenshots:
[first failure](images/first.png), [second failure](images/second.png).

## Docker

From the project root, with Docker available:

```bash
docker build -t fse-ops:0.1 .
docker run --rm fse-ops:0.1
```

The two-stage image builds and installs the wheel, then runs the five tests.
The bug report, unsupported case, and profiling command/result are in
[NOTE.md](NOTE.md).
