# Project note

## Docker small Bug

During the Docker debugging exercise, the image built successfully, but pytest
failed during collection with `ModuleNotFoundError: No module named 'fse_ops'`.
The wheel was installed with system Python (`/usr/bin/python3`), while tests
used `/opt/venv/bin/python3`. The fix installs the wheel with `python3 -m pip`
in the runtime venv and checks `import fse_ops` during the build.

Fix commit: `043124e079a6a14da9fdf95f3f5fa10af20cf9ea`.

## Testing Bug Detection

To verify that the tests can detect incorrect behavior, we intentionally introduced a bug in `src/fse_ops/ops.cpp` by modifying the following line:

**Correct version:**

```cpp
output_row[col] = window[12];
```

**Version with the bug:**

```cpp
output_row[col] = window[11];
```

A 5×5 window contains 25 elements, so the median corresponds to index 12 (using zero-based indexing). Using index 11 instead can produce incorrect results.

After rebuilding the C++ extension, I ran the tests using `pytest`. Two out of five tests failed:

- `test_exact_values`
- `test_mixed_values_and_batch`

**1. Result of test_exact_values**

![test_exact_values failure](images/first.png)

**2. Result of test_mixed_values_and_batch**

![test_mixed_values_and_batch failure](images/second.png)

### Analysis of the Results

The pytest output shows that only the tests checking numerical values failed, while the tests for output shape, data type, and error handling passed.

This suggests that the problem is likely related to the median calculation. Therefore, it makes sense to check how the median element is selected after calling `std::nth_element`.

Fix commit: `6cf76ebde50b686962af5f75c852984da40ae992`.

## Unsupported case

A CPU float32 tensor of shape `(4, 4)` is too small for a 5x5 window.
The C++ check raises `RuntimeError: median_filter5x5: H and W must both be
at least 5`. Other dtypes and GPU tensors are also rejected.

## Profiling

After building the Docker image as in README, run from the repository root
on the host:

```bash
docker run --rm -v "${PWD}/tools:/profile:ro" fse-ops:0.1 python3 -m cProfile -s cumtime /profile/profile_op.py
```

The script uses shape `(4, 3, 512, 512)`, five calls, seed 0, and one thread.
The recorded run on 2026-10-25 measured **4.528 seconds** cumulatively in
`fse_ops._C.median_filter5x5`, or **0.906 seconds per call** at displayed
precision. Output shape was `(4, 3, 508, 508)`. This used Linux x86_64,
Python 3.12.3, and CPU PyTorch 2.10.0. Timings depend on the machine.
