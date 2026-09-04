# -*- coding: utf-8 -*-
"""手续费收益模块的验证脚本。

直接运行: python tests/test_fee_income.py
(不依赖 pytest，全部用 assert；也兼容 pytest 收集)

覆盖:
1. 黄金值回归: fee_apr 缺省时，旧有输出列与改动前 main 分支逐位一致(硬编码基线)
2. 闭式解: 价格恒定(无IL、无对冲)时，收入 = 本金 × APR × 实际经过时长
   (N 行数据只有 N-1 小时时间流逝，首行不计费)
3. 缺失K线: 行间隔 3 小时按 3 小时计费
4. V3 区间开关: 出区间不计费；恰好停在区间边界(单边资产)也不计费
5. 实例复用: 同一实例 run 两次，结果与新实例一致；summary 从传入的
   results 读数，不受实例后续状态污染
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.uniswap_v2_strategy import UniswapV2Strategy
from backtest.uniswap_v3_strategy import UniswapV3Strategy

HOURS_PER_YEAR = 365 * 24

# 改动前 main 分支(1d5ff9f)在下方 make_gbm_data() 固定种子数据上的末行输出，
# 由 git worktree 检出旧代码实跑得到。fee_apr 缺省时新代码必须逐位复现。
GOLDEN_MAIN = {
    "V2": dict(
        unhedged_value=90578.08056259688,
        hedged_value=90266.6795484579,
        lp_value=90578.08056259688,
        impermanent_loss=-0.4876437624638741,
        total_funding_cost=34.79536687080596,
        total_hedge_cost=54.856250596654355,
        cumulative_hedge_pnl=-221.74939667151023,
        hedge_count=98,
    ),
    "V3": dict(
        unhedged_value=86817.6143263411,
        hedged_value=83868.88536187579,
        lp_value=86817.6143263411,
        impermanent_loss=-4.619028015665161,
        total_funding_cost=327.546746510739,
        total_hedge_cost=529.7132719873385,
        cumulative_hedge_pnl=-2091.4689459672536,
        hedge_count=119,
    ),
}


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


def new_v2(**kw):
    return UniswapV2Strategy(
        initial_capital=100000, price_threshold=0.02, funding_rate=0.08, **kw
    )


def new_v3(**kw):
    return UniswapV3Strategy(
        initial_capital=100000,
        price_range_lower=2400,
        price_range_upper=3750,
        price_threshold=0.02,
        funding_rate=0.08,
        **kw,
    )


def test_golden_regression_when_fee_disabled():
    """fee_apr 缺省时: 旧有列与改动前 main 分支逐位一致，新增列全为0"""
    data = make_gbm_data()
    for tag, strat in [("V2", new_v2()), ("V3", new_v3())]:
        r = strat.run(data)
        last = r.iloc[-1]
        g = GOLDEN_MAIN[tag]
        for col in [
            "unhedged_value",
            "hedged_value",
            "lp_value",
            "impermanent_loss",
        ]:
            assert abs(last[col] - g[col]) < 1e-9, (tag, col, last[col], g[col])
        assert abs(strat.total_funding_cost - g["total_funding_cost"]) < 1e-9
        assert abs(strat.total_hedge_cost - g["total_hedge_cost"]) < 1e-9
        assert abs(strat.cumulative_hedge_pnl - g["cumulative_hedge_pnl"]) < 1e-9
        assert strat.hedge_count == g["hedge_count"]
        assert strat.total_fee_income == 0
        assert (r["fee_income"] == 0).all()
        assert (r["unhedged_value"] == r["lp_value"]).all()


def test_flat_price_matches_closed_form():
    """价格不动时: 收入 = 本金 × APR × (N-1)小时/年小时数(首行不计费)"""
    hours = 720
    capital = 100000.0
    fee_apr = 0.12
    data = make_flat_data(hours=hours)
    expected = capital * fee_apr * (hours - 1) / HOURS_PER_YEAR

    v2 = new_v2(fee_apr=fee_apr)
    r2 = v2.run(data)
    assert abs(v2.total_fee_income - expected) < 1e-6, (v2.total_fee_income, expected)
    assert r2.iloc[0]["fee_income"] == 0  # 首行无时间流逝
    # 价值方程: unhedged = lp + fee; 无对冲时 hedged 同
    last = r2.iloc[-1]
    assert abs(last["unhedged_value"] - (last["lp_value"] + v2.total_fee_income)) < 1e-9
    assert abs(last["hedged_value"] - (last["lp_value"] + v2.total_fee_income)) < 1e-9

    v3 = new_v3(fee_apr=fee_apr)
    v3.run(data)
    assert abs(v3.total_fee_income - expected) < 1e-6, (v3.total_fee_income, expected)


def test_gap_in_data_bills_actual_elapsed_hours():
    """缺失K线: 两行相隔3小时按3小时计费(仓位一直在池子里)"""
    capital = 100000.0
    fee_apr = 0.12
    ts = pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 04:00"]
    )  # 第二段缺2根K线
    data = pd.DataFrame({"timestamp": ts, "close": [3000.0] * 3})

    v2 = new_v2(fee_apr=fee_apr)
    v2.run(data)
    expected = capital * fee_apr * (1 + 3) / HOURS_PER_YEAR
    assert abs(v2.total_fee_income - expected) < 1e-9, (v2.total_fee_income, expected)


def test_v3_stops_accruing_out_of_range_and_at_boundary():
    """V3: 出区间不计费；恰好在区间边界(仓位100%单边)也不计费"""
    capital = 100000.0
    fee_apr = 0.24
    upper, lower = 3750.0, 2400.0
    # 100h 区间内 → 50h 恰好在上界 → 100h 跳出上界 → 50h 恰好在下界 → 100h 回区间内
    prices = (
        [3000.0] * 100 + [upper] * 50 + [5000.0] * 100 + [lower] * 50 + [3000.0] * 100
    )
    ts = pd.date_range("2025-01-01", periods=len(prices), freq="h")
    data = pd.DataFrame({"timestamp": ts, "close": prices})

    v3 = new_v3(fee_apr=fee_apr)
    r3 = v3.run(data)

    px = r3["price"]
    boundary_or_out = (px >= upper) | (px <= lower)
    assert boundary_or_out.sum() == 200
    # 边界与出区间的行全不计费
    assert (r3.loc[boundary_or_out, "fee_income"] == 0).all()
    # 这些时段 total_fee_income 平台期零增长(按连续段检查)
    assert r3.iloc[100:250]["total_fee_income"].nunique() == 1
    assert r3.iloc[250:300]["total_fee_income"].nunique() == 1
    # in_range 列保持旧的闭区间展示口径: 恰好在边界仍算 True
    assert bool(r3.iloc[100]["in_range"]) and bool(r3.iloc[250]["in_range"])
    # 计费行 = 严格在区间内的行(去掉首行)
    strictly_in = (px > lower) & (px < upper)
    billed = r3["fee_income"] > 0
    assert (billed == (strictly_in & (r3.index != 0))).all()
    # 累计列与逐行列自洽
    assert abs(r3["fee_income"].sum() - v3.total_fee_income) < 1e-9


def test_instance_reuse_no_cross_contamination():
    """同一实例run两次: 第二次结果与新实例一致; summary跟随传入的results"""
    data_a = make_gbm_data(seed=42)
    data_b = make_gbm_data(seed=7)

    for factory in (new_v2, new_v3):
        reused = factory(fee_apr=0.12)
        results_a = reused.run(data_a)
        summary_a_snapshot = reused.get_summary(results_a)
        results_b = reused.run(data_b)

        fresh = factory(fee_apr=0.12)
        results_b_fresh = fresh.run(data_b)

        # 复用实例的第二次run与新实例逐位一致(无累积污染)
        for col in [
            "unhedged_value",
            "hedged_value",
            "total_fee_income",
            "total_funding_cost",
            "hedge_cost",
        ]:
            assert (results_b[col] == results_b_fresh[col]).all(), (
                factory.__name__,
                col,
            )

        # 第二次run后，用第一次的results再取summary，手续费/成本仍是第一次的数
        summary_a_after = reused.get_summary(results_a)
        assert summary_a_after["手续费收入"] == summary_a_snapshot["手续费收入"]
        assert summary_a_after["资金费用"] == summary_a_snapshot["资金费用"]
        assert summary_a_after["对冲成本"] == summary_a_snapshot["对冲成本"]
        assert summary_a_after["对冲调整次数"] == summary_a_snapshot["对冲调整次数"]


if __name__ == "__main__":
    test_golden_regression_when_fee_disabled()
    print("PASS: fee_apr缺省时与main分支黄金值逐位一致")
    test_flat_price_matches_closed_form()
    print("PASS: 恒定价格下收入与闭式解一致，首行不计费 (V2 & V3)")
    test_gap_in_data_bills_actual_elapsed_hours()
    print("PASS: 缺失K线按实际间隔计费")
    test_v3_stops_accruing_out_of_range_and_at_boundary()
    print("PASS: V3出区间与恰好在边界均不计费")
    test_instance_reuse_no_cross_contamination()
    print("PASS: 实例复用无跨回测污染，summary跟随results")
    print("\n全部通过")
