# -*- coding: utf-8 -*-
"""手续费收益模块的验证脚本。

直接运行: python tests/test_fee_income.py
(不依赖 pytest，全部用 assert；也兼容 pytest 收集)

覆盖三件事:
1. 向后兼容: fee_apr 缺省(=0)时，unhedged_value 等于纯 lp_value，旧行为不变
2. 闭式解: 价格恒定(无IL、无对冲)时，费用收入 == 本金 × fee_apr × 时长
3. V3 区间开关: 价格出区间的小时不计费，只有在区间内的小时累积
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.uniswap_v2_strategy import UniswapV2Strategy
from backtest.uniswap_v3_strategy import UniswapV3Strategy

HOURS_PER_YEAR = 365 * 24


def make_flat_data(price=3000.0, hours=720):
    """恒定价格序列: 无无常损失、不触发对冲，只剩手续费一个变量"""
    ts = pd.date_range("2025-01-01", periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "close": [price] * hours})


def make_gbm_data(start_price=3000.0, hours=1440, vol=0.03, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, vol / np.sqrt(24), hours - 1)
    prices = start_price * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    ts = pd.date_range("2025-01-01", periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "close": prices})


def test_default_zero_fee_keeps_old_behavior():
    """fee_apr 缺省时 unhedged_value == lp_value(逐行), 总费用收入为0"""
    data = make_gbm_data()
    v2 = UniswapV2Strategy(
        initial_capital=100000, price_threshold=0.02, funding_rate=0.08
    )
    r2 = v2.run(data)
    assert v2.total_fee_income == 0
    assert (r2["unhedged_value"] == r2["lp_value"]).all()

    v3 = UniswapV3Strategy(
        initial_capital=100000,
        price_range_lower=2400,
        price_range_upper=3750,
        price_threshold=0.02,
        funding_rate=0.08,
    )
    r3 = v3.run(data)
    assert v3.total_fee_income == 0
    assert (r3["unhedged_value"] == r3["lp_value"]).all()


def test_flat_price_matches_closed_form():
    """价格不动时: 收入 = 本金 × APR × 小时数/年小时数 (逐小时线性累积)"""
    hours = 720
    capital = 100000.0
    fee_apr = 0.12
    data = make_flat_data(hours=hours)

    v2 = UniswapV2Strategy(
        initial_capital=capital,
        price_threshold=0.02,
        funding_rate=0.08,
        fee_apr=fee_apr,
    )
    r2 = v2.run(data)
    expected = capital * fee_apr * hours / HOURS_PER_YEAR
    assert abs(v2.total_fee_income - expected) < 1e-6, (
        v2.total_fee_income,
        expected,
    )
    # 价值方程: unhedged = lp + fee, hedged 同样带上 fee
    last = r2.iloc[-1]
    assert abs(last["unhedged_value"] - (last["lp_value"] + v2.total_fee_income)) < 1e-9
    assert abs(last["hedged_value"] - (last["lp_value"] + v2.total_fee_income)) < 1e-9

    v3 = UniswapV3Strategy(
        initial_capital=capital,
        price_range_lower=2400,
        price_range_upper=3750,
        price_threshold=0.02,
        funding_rate=0.08,
        fee_apr=fee_apr,
    )
    v3.run(data)
    assert abs(v3.total_fee_income - expected) < 1e-6, (
        v3.total_fee_income,
        expected,
    )


def test_v3_stops_accruing_out_of_range():
    """价格出区间的小时不计费; 收入只等于区间内小时的累积"""
    capital = 100000.0
    fee_apr = 0.24
    # 300小时在区间内(3000), 然后跳出上限(5000)停留300小时, 再回到区间内300小时
    prices = [3000.0] * 300 + [5000.0] * 300 + [3000.0] * 300
    ts = pd.date_range("2025-01-01", periods=len(prices), freq="h")
    data = pd.DataFrame({"timestamp": ts, "close": prices})

    v3 = UniswapV3Strategy(
        initial_capital=capital,
        price_range_lower=2400,
        price_range_upper=3750,
        price_threshold=0.02,
        funding_rate=0.08,
        fee_apr=fee_apr,
    )
    r3 = v3.run(data)

    # 出区间的300小时 fee_income 全为0
    out_rows = r3[~r3["in_range"]]
    assert len(out_rows) == 300
    assert (out_rows["fee_income"] == 0).all()

    # 出区间期间 total_fee_income 不增长
    plateau = r3.iloc[300:600]["total_fee_income"]
    assert plateau.nunique() == 1

    # 总收入 = 各小时 lp_value×小时费率 只在区间内求和 (数值上应与逐行列一致)
    assert abs(r3["fee_income"].sum() - v3.total_fee_income) < 1e-9
    # 且严格小于"全程计费"的水平
    full_accrual = r3["lp_value"].mul(fee_apr / HOURS_PER_YEAR).sum()
    assert v3.total_fee_income < full_accrual


if __name__ == "__main__":
    test_default_zero_fee_keeps_old_behavior()
    print("PASS: fee_apr缺省时旧行为不变")
    test_flat_price_matches_closed_form()
    print("PASS: 恒定价格下收入与闭式解一致 (V2 & V3)")
    test_v3_stops_accruing_out_of_range()
    print("PASS: V3出区间停止计费")
    print("\n全部通过")
