# -*- coding: utf-8 -*-
"""手续费收益模块的验证脚本。

直接运行: python tests/test_fee_income.py
(不依赖 pytest，全部用 assert；也兼容 pytest 收集)

覆盖:
1. 黄金值回归: fee_apr 缺省时，全部旧有输出列逐行与改动前 main 分支一致
   (对每列做 sha256 哈希比对，非只看末行)
2. 闭式解: 价格恒定(无IL、无对冲)时，收入 = 本金 × APR × 实际经过时长
   (N 行数据只有 N-1 小时时间流逝，首行不计费)
3. 缺失K线: 行间隔 3 小时按 3 小时计费(以缺口起点状态/价值近似)
4. V3 区间开关: 计费活跃判定下闭上开 [lower, upper)，与 Uniswap tick 语义
   一致；计费归属按区间起点(上一行)状态
5. 乱序时间戳: fee_apr>0 时直接拒绝(否则会静默重复计时)；fee_apr=0 不受影响
6. 非常规索引: 未 reset_index 的输入(如 get_price 缓存切片)结果与默认索引一致
7. 实例复用: 同一实例 run 两次，结果与新实例一致；summary 从传入的
   results 读数，不受实例后续状态污染
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.uniswap_v2_strategy import UniswapV2Strategy
from backtest.uniswap_v3_strategy import UniswapV3Strategy

HOURS_PER_YEAR = 365 * 24

# 改动前 main 分支(1d5ff9f)在 make_gbm_data() 固定种子数据上的全列哈希，
# 由 git worktree 检出旧代码实跑得到(列值 round(8) 后 sha256 前16位)。
# fee_apr 缺省时新代码必须逐列复现。
GOLDEN_COL_HASH = {
    "V2": {
        "price": "3a289db3e2301931",
        "lp_value": "d9380a73061b93bc",
        "hodl_value": "bd487307c8a38dd6",
        "lp_vs_hodl": "c59f8bb2945edeec",
        "hedge_vs_hodl": "86110c8f370c93cf",
        "impermanent_loss": "e8750ededcf39e64",
        "eth_amount": "2cd24d2e56c253f8",
        "usdt_amount": "243b01687f0fd1f6",
        "hedge_position": "df8e651d4bcd1421",
        "hedge_adjusted": "111e338168d3fd05",
        "funding_fee": "1cfd68ba082d2fbc",
        "total_funding_cost": "7f2a3b8432526696",
        "hedge_cost": "c844f7c3cf13365b",
        "hedge_pnl": "4900c25c66b0ca63",
        "cumulative_hedge_pnl": "b8adee89b6724d80",
        "unhedged_value": "d9380a73061b93bc",
        "hedged_value": "feeaaaefe2f8406b",
        "unhedged_return": "1609ac2e9fcd1a2e",
        "hedged_return": "e01b89c4deae72c2",
    },
    "V3": {
        "price": "3a289db3e2301931",
        "in_range": "02a6716f4f5fa18f",
        "lp_value": "91e159a66e9eb4f4",
        "hodl_value": "bd487307c8a38dd6",
        "lp_vs_hodl": "461617f159e84d06",
        "hedge_vs_hodl": "2fe6b0ae6758fb9c",
        "impermanent_loss": "5e1cb4961c1ef7ca",
        "eth_amount": "41acc051568d2ea1",
        "usdt_amount": "137cb3b324f6f5a0",
        "hedge_position": "91b44565bc7987a5",
        "hedge_adjusted": "fd04bc65fc6c565a",
        "funding_fee": "c0ad9ccd8f036a41",
        "total_funding_cost": "97612bce9d3381ef",
        "hedge_cost": "df72864f66e0d118",
        "hedge_pnl": "309cdc1a068e0b0d",
        "cumulative_hedge_pnl": "f66008b75fd8178c",
        "unhedged_value": "91e159a66e9eb4f4",
        "hedged_value": "ce88d7e74a383bd5",
        "unhedged_return": "3f60a77bc3ff1022",
        "hedged_return": "14c6028060a9878a",
    },
}


def col_hash(series):
    arr = np.asarray(series, dtype=float).round(8) + 0.0  # +0.0 归一 -0.0
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


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
    """fee_apr 缺省时: 全部旧有列逐行与 main 分支一致(哈希)，新增列全为0"""
    data = make_gbm_data()
    for tag, strat in [("V2", new_v2()), ("V3", new_v3())]:
        r = strat.run(data)
        for col, expected in GOLDEN_COL_HASH[tag].items():
            got = col_hash(r[col])
            assert got == expected, (tag, col, got, expected)
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


def test_non_monotonic_timestamps_rejected_when_billing():
    """乱序时间戳: fee_apr>0 时报错(避免静默重复计时)；fee_apr=0 保持旧行为"""
    ts = pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 03:00", "2025-01-01 01:00", "2025-01-01 04:00"]
    )
    data = pd.DataFrame({"timestamp": ts, "close": [3000.0] * 4})

    try:
        new_v2(fee_apr=0.12).run(data)
        raise AssertionError("乱序时间戳应当报错")
    except ValueError as e:
        assert "单调" in str(e)

    # 不计费时不校验时间戳，旧行为不变
    r = new_v2().run(data)
    assert len(r) == 4


def test_v3_range_gating_with_boundary_semantics():
    """V3: 计费按区间起点状态；活跃判定下闭上开(恰在下界计费、恰在上界不计费)"""
    capital = 100000.0
    fee_apr = 0.24
    upper, lower = 3750.0, 2400.0
    # 100h 区间内 → 50h 恰好在上界 → 100h 出上界 → 50h 恰好在下界 → 100h 回区间内
    prices = (
        [3000.0] * 100 + [upper] * 50 + [5000.0] * 100 + [lower] * 50 + [3000.0] * 100
    )
    ts = pd.date_range("2025-01-01", periods=len(prices), freq="h")
    data = pd.DataFrame({"timestamp": ts, "close": prices})

    v3 = new_v3(fee_apr=fee_apr)
    r3 = v3.run(data)

    px = r3["price"]
    fee_active = (px >= lower) & (px < upper)  # 下闭上开
    # 计费行 = 上一行处于活跃态(区间起点归属)
    expected_billed = fee_active.shift(1, fill_value=False)
    assert ((r3["fee_income"] > 0) == expected_billed).all()

    # 具体边界语义:
    # 行100(恰到上界): 前一小时仍在区间内 → 该行计费；之后停止
    assert r3.iloc[100]["fee_income"] > 0
    assert (r3.iloc[101:251]["fee_income"] == 0).all()
    # 停费期间累计收入零增长
    assert r3.iloc[100:251]["total_fee_income"].nunique() == 1
    # 行251(上一行恰在下界): 下界是活跃的(Uniswap下闭语义) → 计费
    assert r3.iloc[251]["fee_income"] > 0

    # in_range 列保持旧的闭区间展示口径: 恰在上界/下界仍为 True
    assert bool(r3.iloc[100]["in_range"]) and bool(r3.iloc[250]["in_range"])

    # 累计列与逐行列自洽
    assert abs(r3["fee_income"].sum() - v3.total_fee_income) < 1e-9


def test_non_default_index_input():
    """未reset_index的输入(如get_price缓存切片): 结果与默认索引完全一致"""
    data = make_gbm_data(hours=720)
    shifted = data.copy()
    shifted.index = range(1000, 1000 + len(shifted))  # 模拟过滤后的非零起始索引

    for factory in (new_v2, new_v3):
        r_default = factory(fee_apr=0.12).run(data)
        r_shifted = factory(fee_apr=0.12).run(shifted)
        for col in ["fee_income", "total_fee_income", "hedge_pnl", "hedged_value"]:
            assert (r_default[col] == r_shifted[col]).all(), (factory.__name__, col)


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


def test_summary_tolerates_legacy_results_without_fee_columns():
    """旧版保存的results(无手续费列)仍能生成summary，手续费按0处理"""
    data = make_gbm_data(hours=720)
    for factory in (new_v2, new_v3):
        strat = factory()
        r = strat.run(data)
        legacy = r.drop(columns=["fee_income", "total_fee_income"])
        summary = strat.get_summary(legacy)
        assert summary["手续费收入"] == "0.00 USDT"


if __name__ == "__main__":
    test_golden_regression_when_fee_disabled()
    print("PASS: fee_apr缺省时全部旧有列逐行与main分支一致(哈希)")
    test_flat_price_matches_closed_form()
    print("PASS: 恒定价格下收入与闭式解一致，首行不计费 (V2 & V3)")
    test_gap_in_data_bills_actual_elapsed_hours()
    print("PASS: 缺失K线按实际间隔计费")
    test_non_monotonic_timestamps_rejected_when_billing()
    print("PASS: 乱序时间戳在计费时被拒绝，不计费时不受影响")
    test_v3_range_gating_with_boundary_semantics()
    print("PASS: V3按区间起点状态计费，下闭上开边界语义")
    test_non_default_index_input()
    print("PASS: 非默认索引输入结果一致(reset_index矫正)")
    test_instance_reuse_no_cross_contamination()
    print("PASS: 实例复用无跨回测污染，summary跟随results")
    test_summary_tolerates_legacy_results_without_fee_columns()
    print("PASS: 旧schema的results可生成summary(手续费按0)")
    print("\n全部通过")
