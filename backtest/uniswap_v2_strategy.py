import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


class UniswapV2Strategy:
    def __init__(
        self, initial_capital, price_threshold, funding_rate, gas_cost_per_hedge=0.001
    ):  # 对冲成本费率(交易量的百分比)
        """
        初始化Uniswap V2 LP策略

        Args:
            initial_capital (float): 初始资金(USDT)
            price_threshold (float): 价格波动阈值，触发对冲的百分比(如0.01表示1%)
            funding_rate (float): 年化资金费率(如0.05表示5%)
            gas_cost_per_hedge (float): 对冲交易的费率(交易量的百分比，如0.001表示0.1%)
        """
        self.initial_capital = initial_capital
        self.price_threshold = price_threshold
        self.funding_rate = funding_rate
        self.gas_cost_per_hedge = gas_cost_per_hedge

        # 跟踪指标
        self.total_hedge_cost = 0
        self.total_funding_cost = 0
        self.hedge_count = 0
        self.hedge_history = []
        self.cumulative_hedge_pnl = 0  # 添加累积对冲盈亏跟踪

    def initialize_position(self, initial_price):
        """初始化LP仓位"""
        # Uniswap V2中，资金50%/50%分配到两种资产
        self.initial_eth = self.initial_capital / (2 * initial_price)
        self.initial_usdt = self.initial_capital / 2

        # 计算常数乘积k
        self.k = self.initial_eth * self.initial_usdt

        # 初始化对冲状态
        self.last_hedge_price = initial_price
        self.hedge_position = 0  # 空头仓位数量(ETH)

    def calculate_lp_position(self, current_price):
        """计算当前LP仓位和价值"""
        # 使用常数乘积公式计算当前数量
        eth_amount = np.sqrt(self.k / current_price)
        usdt_amount = np.sqrt(self.k * current_price)

        # LP总价值
        lp_value = eth_amount * current_price + usdt_amount

        return lp_value, eth_amount, usdt_amount

    def calculate_hodl_value(self, current_price):
        """计算简单持有策略的价值"""
        return self.initial_eth * current_price + self.initial_usdt

    def calculate_impermanent_loss(self, current_price):
        """计算无常损失(百分比)"""
        lp_value, _, _ = self.calculate_lp_position(current_price)
        hodl_value = self.calculate_hodl_value(current_price)
        return (lp_value / hodl_value) - 1

    def check_and_adjust_hedge(self, current_price, timestamp):
        """检查并调整对冲仓位"""
        # 计算价格变化百分比
        price_change_pct = abs(current_price / self.last_hedge_price - 1)

        # 如果价格变化超过阈值，调整对冲
        if price_change_pct >= self.price_threshold:
            # 计算LP中的ETH头寸
            _, current_eth, _ = self.calculate_lp_position(current_price)

            # 对冲头寸应该抵消与初始头寸的差异
            new_hedge_position = self.initial_eth - current_eth

            # 记录调整
            hedge_adjustment = new_hedge_position - self.hedge_position

            # 计算对冲交易量并基于交易量计算成本
            hedge_trade_volume = abs(hedge_adjustment) * current_price
            hedge_cost = hedge_trade_volume * self.gas_cost_per_hedge

            self.hedge_position = new_hedge_position

            # 记录成本
            self.total_hedge_cost += hedge_cost
            self.hedge_count += 1

            # 记录历史
            self.hedge_history.append(
                {
                    "timestamp": timestamp,
                    "price": current_price,
                    "hedge_adjustment": hedge_adjustment,
                    "new_position": new_hedge_position,
                    "trade_volume": hedge_trade_volume,
                    "hedge_cost": hedge_cost,
                }
            )

            # 更新参考价格
            self.last_hedge_price = current_price

            return True

        return False

    def calculate_funding_fee(self, hours_passed):
        """计算资金费用"""
        if self.hedge_position == 0:
            return 0

        # 将年化费率转换为小时费率
        hourly_funding_rate = self.funding_rate / (365 * 24)

        # 计算资金费用
        funding_fee = (
            abs(self.hedge_position)
            * self.last_hedge_price
            * hourly_funding_rate
            * hours_passed
        )

        self.total_funding_cost += funding_fee
        return funding_fee

    def run(self, data):
        """运行回测"""
        if len(data) == 0:
            return pd.DataFrame()

        # 用第一个价格初始化仓位
        initial_price = data.iloc[0]["close"]
        self.initialize_position(initial_price)

        # 重置累积对冲盈亏
        self.cumulative_hedge_pnl = 0

        results = []

        for i, row in data.iterrows():
            timestamp = row["timestamp"]
            current_price = row["close"]

            # 计算LP仓位价值
            lp_value, eth_amount, usdt_amount = self.calculate_lp_position(
                current_price
            )

            # 计算HODL价值
            hodl_value = self.calculate_hodl_value(current_price)

            # 检查并调整对冲
            hedge_adjusted = self.check_and_adjust_hedge(current_price, timestamp)

            # 计算资金费用(假设每条数据间隔1小时)
            hours_passed = 1
            funding_fee = self.calculate_funding_fee(hours_passed)

            # 计算对冲PnL
            hedge_pnl = 0
            if i > 0:
                previous_price = data.iloc[i - 1]["close"]
                # 获取上一个时刻的hedge_position
                previous_hedge_position = self.hedge_position
                if hedge_adjusted and len(self.hedge_history) > 0:
                    # 如果当前时刻有对冲调整，使用调整前的仓位
                    previous_hedge_position = (
                        self.hedge_position - self.hedge_history[-1]["hedge_adjustment"]
                    )
                # 空头仓位在价格下跌时获利
                hedge_pnl = previous_hedge_position * (current_price - previous_price)

                # 更新累积对冲盈亏
                self.cumulative_hedge_pnl += hedge_pnl

            # 计算无常损失
            il = self.calculate_impermanent_loss(current_price)

            # 计算总价值 (不考虑手续费)
            unhedged_value = lp_value
            hedged_value = (
                lp_value
                + self.cumulative_hedge_pnl
                - self.total_hedge_cost
                - self.total_funding_cost
            )

            # 记录结果
            results.append(
                {
                    "timestamp": timestamp,
                    "price": current_price,
                    "lp_value": lp_value,
                    "hodl_value": hodl_value,
                    "lp_vs_hodl": lp_value - hodl_value,  # LP策略与HODL策略的差额
                    "hedge_vs_hodl": hedged_value
                    - hodl_value,  # 对冲策略与HODL策略的差额
                    "impermanent_loss": il * 100,  # 转换为百分比
                    "eth_amount": eth_amount,
                    "usdt_amount": usdt_amount,
                    "hedge_position": self.hedge_position,
                    "hedge_adjusted": hedge_adjusted,
                    "funding_fee": funding_fee,
                    "total_funding_cost": self.total_funding_cost,
                    "hedge_cost": self.total_hedge_cost,
                    "hedge_pnl": hedge_pnl,
                    "cumulative_hedge_pnl": self.cumulative_hedge_pnl,
                    "unhedged_value": unhedged_value,
                    "hedged_value": hedged_value,
                    "unhedged_return": (unhedged_value / self.initial_capital - 1)
                    * 100,  # 百分比
                    "hedged_return": (hedged_value / self.initial_capital - 1)
                    * 100,  # 百分比
                }
            )

        return pd.DataFrame(results)

    def get_summary(self, results):
        """生成回测结果摘要"""
        if len(results) == 0:
            return {}

        first_row = results.iloc[0]
        last_row = results.iloc[-1]

        # 计算对冲相对HODL收益率
        hedge_vs_hodl_return = (
            last_row["hedged_value"] / last_row["hodl_value"] - 1
        ) * 100

        # 计算最终增益
        final_gain = hedge_vs_hodl_return - last_row["impermanent_loss"]

        # 计算回测总小时数（假设数据是按小时记录的）
        if isinstance(first_row["timestamp"], str):
            start_time = pd.to_datetime(first_row["timestamp"])
            end_time = pd.to_datetime(last_row["timestamp"])
        else:
            start_time = first_row["timestamp"]
            end_time = last_row["timestamp"]

        total_hours = (end_time - start_time).total_seconds() / 3600
        years = total_hours / (365 * 24)

        # 计算年化率
        def annualize(rate):
            # 将百分比转换为小数
            rate_decimal = rate / 100
            # 计算年化率
            if years > 0:
                return ((1 + rate_decimal) ** (1 / years) - 1) * 100
            return rate

        hedge_vs_hodl_annual = annualize(hedge_vs_hodl_return)
        impermanent_loss_annual = annualize(last_row["impermanent_loss"])
        final_gain_annual = annualize(final_gain)

        return {
            "初始资金": self.initial_capital,
            "时间段": f"{first_row['timestamp']} 至 {last_row['timestamp']}",
            "资金费用": f"{self.total_funding_cost:.2f} USDT",
            "对冲成本": f"{self.total_hedge_cost:.2f} USDT",
            "对冲调整次数": self.hedge_count,
            "ETH价格变化": f"{(last_row['price'] / first_row['price'] - 1) * 100:.2f}%",
            "不对冲收益率": f"{last_row['unhedged_return']:.2f}%",
            "对冲收益率": f"{last_row['hedged_return']:.2f}%",
            "对冲相对HODL收益率": f"{hedge_vs_hodl_return:.2f}%",
            "最终无常损失": f"{last_row['impermanent_loss']:.2f}%",
            "最终增益": f"{final_gain:.2f}%",
            # 年化指标
            "对冲相对HODL收益率(年化)": f"{hedge_vs_hodl_annual:.2f}%/年",
            "最终无常损失(年化)": f"{impermanent_loss_annual:.2f}%/年",
            "最终增益(年化)": f"{final_gain_annual:.2f}%/年",
        }

    def plot_results(self, results, save_path=None):
        """绘制回测结果图表"""
        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 转换时间戳为datetime格式(如果是字符串)
        if isinstance(results["timestamp"].iloc[0], str):
            results["timestamp"] = pd.to_datetime(results["timestamp"])

        plt.figure(figsize=(15, 15))  # 增加高度以容纳四个子图

        # 图1: 价格与对冲调整
        plt.subplot(4, 1, 1)
        plt.plot(results["timestamp"], results["price"], label="ETH价格", color="black")

        # 创建阈值线段
        if len(self.hedge_history) > 0:
            # 将对冲历史按时间排序
            sorted_history = sorted(
                self.hedge_history,
                key=lambda x: (
                    pd.to_datetime(x["timestamp"])
                    if isinstance(x["timestamp"], str)
                    else x["timestamp"]
                ),
            )

            # 获取所有时间点，包括开始、每次对冲和结束
            time_points = [results["timestamp"].iloc[0]]  # 开始时间
            prices = [results["price"].iloc[0]]  # 开始价格

            # 添加所有对冲时间点
            for hedge in sorted_history:
                hedge_time = (
                    pd.to_datetime(hedge["timestamp"])
                    if isinstance(hedge["timestamp"], str)
                    else hedge["timestamp"]
                )
                time_points.append(hedge_time)
                prices.append(hedge["price"])

            # 添加结束时间
            time_points.append(results["timestamp"].iloc[-1])

            # 绘制每个时间段的阈值线
            for i in range(len(time_points) - 1):
                ref_price = prices[i]  # 当前段的参考价格

                # 计算上下阈值
                upper = ref_price * (1 + self.price_threshold)
                lower = ref_price * (1 - self.price_threshold)

                # 绘制当前时间段的阈值线
                plt.hlines(
                    y=upper,
                    xmin=time_points[i],
                    xmax=time_points[i + 1],
                    colors="r",
                    linestyles=":",
                    alpha=0.5,
                    label="上阈值" if i == 0 else "",
                )
                plt.hlines(
                    y=lower,
                    xmin=time_points[i],
                    xmax=time_points[i + 1],
                    colors="g",
                    linestyles=":",
                    alpha=0.5,
                    label="下阈值" if i == 0 else "",
                )

            # 标记对冲点
            for hedge in sorted_history:
                hedge_time = (
                    pd.to_datetime(hedge["timestamp"])
                    if isinstance(hedge["timestamp"], str)
                    else hedge["timestamp"]
                )
                plt.axvline(
                    x=hedge_time,
                    color="y",
                    linestyle="-",
                    alpha=0.3,
                    label="对冲调整" if hedge == sorted_history[0] else "",
                )
        else:
            # 如果没有对冲历史，就只显示基于初始价格的阈值线
            initial_price = results["price"].iloc[0]
            plt.hlines(
                y=initial_price * (1 + self.price_threshold),
                xmin=results["timestamp"].iloc[0],
                xmax=results["timestamp"].iloc[-1],
                colors="r",
                linestyles=":",
                alpha=0.5,
                label="上阈值",
            )
            plt.hlines(
                y=initial_price * (1 - self.price_threshold),
                xmin=results["timestamp"].iloc[0],
                xmax=results["timestamp"].iloc[-1],
                colors="g",
                linestyles=":",
                alpha=0.5,
                label="下阈值",
            )

        plt.title(f"ETH价格和动态对冲阈值 (阈值±{self.price_threshold*100}%)")
        plt.ylabel("价格 (USDT)")
        plt.legend()
        plt.grid(True)

        # 图2: 收益率对比
        plt.subplot(4, 1, 2)
        plt.plot(
            results["timestamp"], results["unhedged_return"], label="不对冲收益率 (%)"
        )
        plt.plot(results["timestamp"], results["hedged_return"], label="对冲收益率 (%)")
        hodl_return = (results["hodl_value"] / self.initial_capital - 1) * 100
        plt.plot(results["timestamp"], hodl_return, label="HODL收益率 (%)")
        plt.title("收益率对比")
        plt.ylabel("收益率 (%)")
        plt.legend()
        plt.grid(True)

        # 图3: 超额收益率对比 (相对于HODL)
        plt.subplot(4, 1, 3)
        hodl_return = (results["hodl_value"] / self.initial_capital - 1) * 100
        # 计算超额收益率
        unhedged_excess = results["unhedged_return"] - hodl_return
        hedged_excess = results["hedged_return"] - hodl_return

        plt.plot(
            results["timestamp"],
            unhedged_excess,
            label="不对冲超额收益率 (%)",
            color="blue",
        )
        plt.plot(
            results["timestamp"], hedged_excess, label="对冲超额收益率 (%)", color="red"
        )
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)  # 零线

        plt.title("超额收益率对比 (相对于HODL)")
        plt.ylabel("超额收益率 (%)")
        plt.legend()
        plt.grid(True)

        # 图4: 成本
        plt.subplot(4, 1, 4)
        plt.plot(results["timestamp"], results["total_funding_cost"], label="资金费用")
        plt.plot(results["timestamp"], results["hedge_cost"], label="对冲成本")
        plt.title("对冲相关成本")
        plt.ylabel("USDT")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 如果指定了保存路径，则保存图表
        if save_path:
            plt.savefig(
                f"{save_path}/backtest_results.png", dpi=300, bbox_inches="tight"
            )

        # plt.show()

    def save_results(self, results, save_dir):
        """保存回测结果和对冲历史"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存对冲历史
        hedge_history_df = pd.DataFrame(self.hedge_history)
        if not hedge_history_df.empty:
            hedge_history_df.to_csv(
                f"{save_dir}/hedge_history_{timestamp}.csv", index=False
            )

        # 保存回测结果
        results.to_csv(f"{save_dir}/backtest_results_{timestamp}.csv", index=False)

        # 保存图表
        self.plot_results(results, save_path=save_dir)

        # 保存摘要
        summary = self.get_summary(results)
        with open(f"{save_dir}/summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print(f"结果已保存到目录: {save_dir}")

    def plot_threshold_comparison(self, results_dict, save_path=None):
        """绘制不同对冲阈值的对比图"""
        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        plt.figure(figsize=(10, 7))

        # 图1: 价格与不同阈值的对冲调整
        plt.subplot(3, 1, 1)
        first_results = results_dict[list(results_dict.keys())[0]]["results"]
        plt.plot(
            first_results["timestamp"],
            first_results["price"],
            label="ETH价格",
            color="black",
        )

        # 为每个阈值绘制对冲调整点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))
        for (threshold, results), color in zip(results_dict.items(), colors):
            strategy = results["strategy"]
            for hedge in strategy.hedge_history:
                if isinstance(hedge["timestamp"], str):
                    hedge_time = pd.to_datetime(hedge["timestamp"])
                else:
                    hedge_time = hedge["timestamp"]
                plt.axvline(
                    x=hedge_time,
                    color=color,
                    linestyle="--",
                    alpha=0.3,
                    label=(
                        f"{threshold*100}%"
                        if hedge == strategy.hedge_history[0]
                        else ""
                    ),
                )

        plt.title("ETH价格和不同阈值的对冲调整")
        plt.ylabel("价格 (USDT)")
        plt.legend()
        plt.grid(True)

        # 图2: 不同阈值的收益率对比
        plt.subplot(3, 1, 2)
        for (threshold, results), color in zip(results_dict.items(), colors):
            plt.plot(
                results["results"]["timestamp"],
                results["results"]["hedged_return"],
                label=f"{threshold*100}%阈值收益率 (%)",
                color=color,
            )

        # 添加HODL收益率作为基准
        hodl_return = (first_results["hodl_value"] / self.initial_capital - 1) * 100
        plt.plot(
            first_results["timestamp"],
            hodl_return,
            label="HODL收益率 (%)",
            color="black",
            linestyle=":",
        )

        plt.title("不同对冲阈值的收益率对比")
        plt.ylabel("收益率 (%)")
        plt.legend()
        plt.grid(True)

        # 图3: 不同阈值的成本对比
        plt.subplot(3, 1, 3)
        for (threshold, results), color in zip(results_dict.items(), colors):
            plt.plot(
                results["results"]["timestamp"],
                results["results"]["total_funding_cost"],
                label=f"{threshold*100}%阈值资金费用",
                color=color,
                linestyle="-",
            )
            plt.plot(
                results["results"]["timestamp"],
                results["results"]["hedge_cost"],
                label=f"{threshold*100}%阈值对冲成本",
                color=color,
                linestyle="--",
            )

        plt.title("不同对冲阈值的成本对比")
        plt.ylabel("USDT")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 如果指定了保存路径，则保存图表
        if save_path:
            plt.savefig(
                f"{save_path}/threshold_comparison.png", dpi=300, bbox_inches="tight"
            )

        plt.show()


def generate_price_data(
    start_price, start_date, end_date, volatility=0.02, drift=0.0001
):
    """
    生成合成价格数据用于测试

    Args:
        start_price (float): 起始价格
        start_date (str): 开始日期，格式'YYYY-MM-DD'
        end_date (str): 结束日期，格式'YYYY-MM-DD'
        volatility (float): 日波动率
        drift (float): 平均日回报率

    Returns:
        pd.DataFrame: 合成价格数据
    """
    # 生成日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq="H")
    hours = len(dates)

    # 生成价格(几何布朗运动)
    prices = [start_price]
    for i in range(1, hours):
        hourly_vol = volatility / np.sqrt(24)
        hourly_drift = drift / 24
        random_return = np.random.normal(hourly_drift, hourly_vol)
        prices.append(prices[-1] * np.exp(random_return))

    return pd.DataFrame({"timestamp": dates, "close": prices})


# 使用示例
if __name__ == "__main__":
    # 初始化参数
    initial_capital = 100000  # 10万USDT
    price_threshold = 0.01  # 2%价格波动触发对冲
    funding_rate = 0.00001  # 5%年化资金费率

    # 创建保存目录
    save_dir = "backtest_results"
    os.makedirs(save_dir, exist_ok=True)

    # 生成测试数据 (2025-2-1 到 2025-4-1)
    data = generate_price_data(
        start_price=3500,
        start_date="2025-02-01",
        end_date="2025-04-01",
        volatility=0.02,  # 日波动率
        drift=0.0001,  # 平均日收益率
    )

    # 创建策略实例
    strategy = UniswapV2Strategy(
        initial_capital=initial_capital,
        price_threshold=price_threshold,
        funding_rate=funding_rate,
        gas_cost_per_hedge=0.001,  # 对冲成本为交易量的0.1%
    )

    # 运行回测
    results = strategy.run(data)

    # 保存结果
    strategy.save_results(results, save_dir)

    # 打印摘要
    summary = strategy.get_summary(results)
    print("\n回测结果摘要:")
    for key, value in summary.items():
        print(f"{key}: {value}")
