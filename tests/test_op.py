
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


def test_output_shape():
    x = torch.randn(4, 3, 10, 12)
    result = median_filter5x5(x)

    assert result.shape == (4, 3, 6, 8)


def test_output_dtype():
    x = torch.randn(7, 7, dtype=torch.float32)
    result = median_filter5x5(x)

    assert result.dtype == torch.float32


def test_exact_values():
    x = torch.arange(49, dtype=torch.float32).reshape(7, 7)

    result = median_filter5x5(x)
    expected = reference(x)

    assert torch.equal(result, expected)


def test_mixed_values_and_batch():
    x = torch.tensor(
        [
            [
                [20, -5, 3, 8, 1, 12],
                [4, 15, -8, 2, 7, 9],
                [6, -3, 11, 0, 14, 5],
                [10, 1, -2, 18, 4, 13],
                [7, 16, 9, -6, 2, 3],
                [8, 0, 12, 5, -4, 17],
            ]
        ],
        dtype=torch.float32,
    ).repeat(4, 1, 1)

    result = median_filter5x5(x)
    expected = reference(x)

    assert result.shape == (4, 2, 2)
    assert torch.equal(result, expected)
    assert not torch.equal(result, x[..., :2, :2])
    assert torch.unique(result).numel() > 1


def test_invalid_dtype():
    x = torch.ones(7, 7, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="float32"):
        median_filter5x5(x)

