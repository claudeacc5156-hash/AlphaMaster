"""Speed-path equivalence: the fast EMA recursion and the vectorised turnover
count must give exactly the same results as the original Python loops."""
import math

import pytest
import torch

from model_core.backtest import MT5Backtest
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


def test_ema_fast_path_propagates_nan_like_loop():
    pytest.importorskip("scipy")
    x = torch.randn(1, 400)
    x[0, 10] = float("nan")
    fast = _ema_recursion_lfilter(x, 2.0 / 6.0)
    ref = _ema_recursion_loop(x, 2.0 / 6.0)
    assert torch.equal(torch.isnan(fast), torch.isnan(ref))
    assert torch.equal(torch.nan_to_num(fast, nan=0.0), torch.nan_to_num(ref, nan=0.0))


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
@pytest.mark.parametrize("pos", [0, 1, 200])
def test_ema_with_inf_matches_loop(bad, pos):
    x = torch.randn(2, 400)
    x[1, pos] = bad
    out = _ema_simple(x, 5)
    ref = _ema_recursion_loop(x, 2.0 / 6.0)
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    assert torch.equal(torch.nan_to_num(out, nan=0.0), torch.nan_to_num(ref, nan=0.0))


def test_ema_fast_path_keeps_loop_memory_layout():
    pytest.importorskip("scipy")
    base = torch.randn(1, 300).cumsum(1)
    for x in (base.expand(6, 300), torch.randn(300, 4).t(), base.unfold(1, 4, 1)[0].t()):
        fast = _ema_recursion_lfilter(x, 2.0 / 21.0)
        ref = _ema_recursion_loop(x, 2.0 / 21.0)
        assert fast.stride() == ref.stride()
        assert torch.equal(fast, ref)
        fast.resize_(fast.numel())  # a normal, resizable torch tensor


def test_ema_falls_back_for_unsupported_inputs():
    x = torch.randn(1, 50, requires_grad=True)
    assert _ema_recursion_lfilter(x, 0.3) is None
    assert _ema_recursion_lfilter(torch.randn(1, 50).half(), 0.3) is None
    neg = torch._neg_view(torch.randn(1, 50))
    assert _ema_recursion_lfilter(neg, 0.3) is None
    assert torch.equal(_ema_simple(neg, 5), _ema_recursion_loop(neg, 2.0 / 6.0))


def _turnover_quality_reference(position: torch.Tensor) -> float:
    """The original per-bar loop, kept here as the reference."""
    N, T = position.shape
    pos_2d = position.tolist()
    all_runs, total_trades = [], 0
    for n in range(N):
        runs, cur_len, cur_dir = [], 0, 0
        for p in pos_2d[n]:
            pi = int(p)
            if pi != 0:
                if pi == cur_dir:
                    cur_len += 1
                else:
                    if cur_len > 0: runs.append(cur_len)
                    cur_dir, cur_len = pi, 1
            else:
                if cur_len > 0: runs.append(cur_len)
                cur_dir, cur_len = 0, 0
        if cur_len > 0: runs.append(cur_len)
        all_runs.extend(runs)
        total_trades += len(runs)
    total_bars = N * T
    target_trades = total_bars / 12.0
    actual_ratio = total_trades / max(target_trades, 1.0)
    if actual_ratio <= 0:
        freq_score = -2.0
    elif actual_ratio < 0.05:
        freq_score = -2.0 + actual_ratio / 0.05
    elif actual_ratio < 0.5:
        freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
    elif actual_ratio <= 2.0:
        log_r = math.log(actual_ratio) / math.log(2.0)
        freq_score = 1.0 * math.exp(-0.5 * log_r ** 2)
    elif actual_ratio <= 8.0:
        freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
    else:
        freq_score = -2.0
    hold_bonus = 0.0
    if all_runs:
        avg_hold = sum(all_runs) / len(all_runs)
        hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)
    return float(freq_score + hold_bonus)


def test_turnover_quality_vectorised_matches_loop():
    bt = MT5Backtest()
    g = torch.Generator().manual_seed(7)
    cases = [
        torch.tanh(torch.randn(1, 3000, generator=g) * 3),           # continuous, |p| < 1
        torch.randint(-1, 2, (1, 3000), generator=g).float(),        # {-1, 0, 1}
        torch.randint(-1, 2, (3, 500), generator=g).float(),         # several rows
        torch.randn(2, 400, generator=g) * 2.5,                      # |p| >= 1 truncates
        torch.ones(1, 100), -torch.ones(1, 100), torch.zeros(1, 100),
        torch.tensor([[1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.9, -0.0, 2.0, 2.7, 1.0]]),
    ]
    sparse = torch.zeros(1, 2400)
    sparse[0, ::40] = 1.0
    cases.append(sparse)
    cases += [torch.tensor([[True, True, False, True, False, False, True]]),
              torch.randint(-2, 3, (2, 300), generator=g)]                  # bool / int dtypes
    for pos in cases:
        assert bt._turnover_quality(pos) == _turnover_quality_reference(pos)
    with pytest.raises(ValueError):
        bt._turnover_quality(torch.tensor([[1.0, float("nan")]]))
