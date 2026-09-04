from get_price import fetch_price_data
from backtest.uniswap_v3_strategy import UniswapV3Strategy
from backtest.uniswap_v2_strategy import UniswapV2Strategy
import pandas as pd


def run_pipeline():
    # 参数 & 抓取（若本地无数据则自动抓取）
    symbol = "APT/USDT"
    start_date = "2025-04-01"
    end_date = "2025-04-10"
    timeframe = "1h"

    df = fetch_price_data(symbol, start_date, end_date, timeframe)
    initial_price = df.iloc[0]["close"]

    # 回测 uni v2 策略
    strat = UniswapV2Strategy(
        initial_capital=1000,
        price_threshold=0.05,
        funding_rate=0.1,
        fee_apr=0.10,  # LP手续费年化率，按目标池子费率APR填(日成交量×费率/TVL×365)
    )
    results = strat.run(df)
    strat.save_results(results, "v2_backtest_results")

    # 回测 uni v3 策略
    strat = UniswapV3Strategy(
        initial_capital=1000,
        price_range_lower=initial_price * 0.5,
        price_range_upper=initial_price * 2,
        price_threshold=0.05,
        funding_rate=0.1,
        gas_cost_per_hedge=0.0003,
        fee_apr=0.20,  # 区间内的手续费年化率(集中流动性放大后)，出区间自动停止计费
    )
    results = strat.run(df)
    strat.save_results(results, "v3_backtest_results")


if __name__ == "__main__":
    run_pipeline()
