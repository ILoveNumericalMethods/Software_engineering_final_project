
import pytest
import torch

from fse_ops import median_filter5x5


def reference(x):
    return (
        x.unfold(-2, 5, 1)
        .unfold(-2, 5, 1)
        .flatten(-2)
        .median(dim=-1)
        .values
    )


# Category 1: Output shape

def test_output_shape_2d():
    x = torch.randn(10, 12)

    result = median_filter5x5(x)

    assert result.shape == (6, 8)


def test_output_shape_batch():
    x = torch.randn(4, 3, 10, 12)

    result = median_filter5x5(x)

    assert result.shape == (4, 3, 6, 8)


# Category 2: Output dtype

def test_output_dtype():
    x = torch.randn(7, 7, dtype=torch.float32)

    result = median_filter5x5(x)

    assert result.dtype == torch.float32


def test_output_dtype_noncontiguous():
    x = torch.randn(10, 12, dtype=torch.float32).T

    assert not x.is_contiguous()

    result = median_filter5x5(x)

    assert result.dtype == x.dtype
    assert torch.equal(result, reference(x))


# Category 3: Exact numerical values

def test_exact_values():
    x = torch.arange(49, dtype=torch.float32).reshape(7, 7)

    result = median_filter5x5(x)
    expected = reference(x)

    assert torch.equal(result, expected)


def test_exact_values_random():
    torch.manual_seed(42)
    x = torch.randn(4, 3, 12, 12)

    result = median_filter5x5(x)
    expected = reference(x)

    assert torch.equal(result, expected)


# Category 4: Detect incorrect implementations

def test_mixed_values_and_batch():
    x = torch.tensor(
        [
            [20, -5, 3, 8, 1, 12],
            [4, 15, -8, 2, 7, 9],
            [6, -3, 11, 0, 14, 5],
            [10, 1, -2, 18, 4, 13],
            [7, 16, 9, -6, 2, 3],
            [8, 0, 12, 5, -4, 17],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(4, 1, 1)

    result = median_filter5x5(x)
    expected = reference(x)

    assert torch.equal(result, expected)
    assert not torch.equal(result, x[..., :2, :2])
    assert torch.unique(result).numel() > 1


def test_impulse_noise():
    x = torch.ones(7, 7, dtype=torch.float32)
    x[1, 1] = 1000.0
    x[3, 3] = -1000.0
    x[5, 5] = 500.0

    result = median_filter5x5(x)
    expected = reference(x)

    assert torch.equal(result, expected)
    assert torch.equal(result, torch.ones(3, 3))
    assert not torch.equal(result, x[:3, :3])


# Category 5: Error handling

def test_invalid_dtype():
    x = torch.ones(7, 7, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="float32"):
        median_filter5x5(x)


def test_input_too_small():
    x = torch.ones(4, 4, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="at least 5"):
        median_filter5x5(x)

