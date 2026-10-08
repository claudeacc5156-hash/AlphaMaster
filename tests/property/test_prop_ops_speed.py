"""Speed-path equivalence: the fast EMA recursion must give exactly the same
results as the original Python loop."""
import pytest
import torch

from model_core.ops import _ema_recursion_lfilter, _ema_recursion_loop, _ema_simple


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("span", [5, 20])
def test_ema_fast_path_is_bit_identical(dtype, span):
    pytest.importorskip("scipy")
    alpha = 2.0 / (span + 1.0)
    g = torch.Generator().manual_seed(span)
    cases = [
        torch.randn(1, 5000, generator=g, dtype=dtype),
        torch.randn(1, 5000, generator=g, dtype=dtype).cumsum(1) * 1e3,
        torch.randn(3, 777, generator=g, dtype=dtype),
        torch.randn(2, 1, generator=g, dtype=dtype),
        torch.randn(2, 2, generator=g, dtype=dtype),
        torch.randn(4, 300, generator=g, dtype=dtype)[:, ::3],  # non-contiguous view
    ]
    for x in cases:
        fast = _ema_recursion_lfilter(x, alpha)
        assert fast is not None
        ref = _ema_recursion_loop(x, alpha)
        assert fast.dtype == ref.dtype and fast.shape == ref.shape
        assert torch.equal(fast, ref)
        assert torch.equal(_ema_simple(x, span), ref)


def test_ema_fast_path_propagates_nan_and_inf_like_loop():
    pytest.importorskip("scipy")
    x = torch.randn(1, 400)
    x[0, 10] = float("nan")
    x[0, 200] = float("inf")
    fast = _ema_recursion_lfilter(x, 2.0 / 6.0)
    ref = _ema_recursion_loop(x, 2.0 / 6.0)
    assert torch.equal(torch.isnan(fast), torch.isnan(ref))
    assert torch.equal(torch.nan_to_num(fast, nan=0.0), torch.nan_to_num(ref, nan=0.0))


def test_ema_falls_back_for_unsupported_inputs():
    x = torch.randn(1, 50, requires_grad=True)
    assert _ema_recursion_lfilter(x, 0.3) is None
    assert _ema_recursion_lfilter(torch.randn(1, 50).half(), 0.3) is None
