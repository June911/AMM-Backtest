import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


class UniswapV3Strategy:
    def __init__(
        self,
        initial_capital,
        price_range_lower,
        price_range_upper,
        price_threshold,
        funding_rate,
        gas_cost_per_hedge=0.001,
    ):
        """
        初始化Uniswap V3 LP策略

        Args:
            initial_capital (float): 初始资金(USDT)
            price_range_lower (float): 集中流动性价格区间下限
            price_range_upper (float): 集中流动性价格区间上限
            price_threshold (float): 价格波动阈值，触发对冲的百分比(如0.01表示1%)
            funding_rate (float): 年化资金费率(如0.05表示5%)
            gas_cost_per_hedge (float): 对冲交易的费率(交易量的百分比，如0.001表示0.1%)
        """
        self.initial_capital = initial_capital
        self.price_range_lower = price_range_lower
        self.price_range_upper = price_range_upper
        self.price_threshold = price_threshold
        self.funding_rate = funding_rate
        self.gas_cost_per_hedge = gas_cost_per_hedge

        # 跟踪指标
        self.total_hedge_cost = 0
        self.total_funding_cost = 0
        self.hedge_count = 0
        self.hedge_history = []
        self.cumulative_hedge_pnl = 0

    def initialize_position(self, initial_price):
        """初始化LP仓位"""
        # 确认初始价格是否在设定的价格区间内
        if (
            initial_price < self.price_range_lower
            or initial_price > self.price_range_upper
        ):
            raise ValueError(
                f"初始价格 {initial_price} 不在设定的价格区间 [{self.price_range_lower}, {self.price_range_upper}] 内"
            )

        # 计算价格范围比例
        self.price_ratio = self.price_range_upper / self.price_range_lower
        self.sqrt_price_ratio = np.sqrt(self.price_ratio)

        # 计算虚拟储备和流动性参数
        # 在 V3 中，L 是常数，表示流动性
        # 计算流动性L，使用完整公式：L = V / [(√P - √P_a) + (1/√P - 1/√P_b)·P]
        # 其中P是当前价格，P_a是下边界价格，P_b是上边界价格
        current_price = initial_price  # 使用初始价格作为当前价格
        usdc_part = np.sqrt(current_price) - np.sqrt(self.price_range_lower)
        eth_part = (
            1 / np.sqrt(current_price) - 1 / np.sqrt(self.price_range_upper)
        ) * current_price
        self.L = self.initial_capital / (usdc_part + eth_part)
        print(self.initial_capital)
        print(f"流动性L: {self.L}")

        # 初始资产数量
        self.initial_eth = self.L * (
            1 / np.sqrt(initial_price) - 1 / np.sqrt(self.price_range_upper)
        )
        self.initial_usdt = self.L * (
            np.sqrt(initial_price) - np.sqrt(self.price_range_lower)
        )
        print(f"初始ETH数量: {self.initial_eth}")
        print(f"初始USDT数量: {self.initial_usdt}")

        # 记录初始值
        self.initial_token0_virtual = self.L / np.sqrt(self.price_range_lower)
        self.initial_token1_virtual = self.L * np.sqrt(self.price_range_upper)

        # 初始化对冲状态
        self.last_hedge_price = initial_price
        self.hedge_position = 0  # 空头仓位数量(ETH)

    def calculate_lp_position(self, current_price):
        """计算当前LP仓位和价值"""
        # V3中，当价格超出区间时，LP持有100%单一资产
        if current_price <= self.price_range_lower:
            # 价格低于下限，100%持有ETH
            eth_amount = self.L * (
                1 / np.sqrt(self.price_range_lower)
                - 1 / np.sqrt(self.price_range_upper)
            )
            usdt_amount = 0
        elif current_price >= self.price_range_upper:
            # 价格高于上限，100%持有USDT
            eth_amount = 0
            usdt_amount = self.L * (
                np.sqrt(self.price_range_upper) - np.sqrt(self.price_range_lower)
            )
        else:
            # 价格在区间内，使用集中流动性公式
            eth_amount = self.L * (
                1 / np.sqrt(current_price) - 1 / np.sqrt(self.price_range_upper)
            )
            usdt_amount = self.L * (
                np.sqrt(current_price) - np.sqrt(self.price_range_lower)
            )

        # LP总价值
        lp_value = eth_amount * current_price + usdt_amount

        return lp_value, eth_amount, usdt_amount

    def calculate_hodl_value(self, current_price):
        """计算简单持有策略的价值"""
        # 计算初始持币策略的当前价值
        initial_eth_value = self.initial_eth * current_price
        return initial_eth_value + self.initial_usdt

    def calculate_impermanent_loss(self, current_price):
        """计算无常损失(百分比)"""
        lp_value, _, _ = self.calculate_lp_position(current_price)
        hodl_value = self.calculate_hodl_value(current_price)
        return (lp_value / hodl_value) - 1

    def check_and_adjust_hedge(self, current_price, timestamp):
        """检查并调整对冲仓位"""
        # 计算价格变化百分比
        price_change_pct = abs(current_price / self.last_hedge_price - 1)

        # 如果价格变化超过阈值或价格跨越了区间边界，调整对冲
        price_crossed_boundary = (
            (
                self.last_hedge_price < self.price_range_lower
                and current_price >= self.price_range_lower
            )
            or (
                self.last_hedge_price > self.price_range_upper
                and current_price <= self.price_range_upper
            )
            or (
                self.last_hedge_price >= self.price_range_lower
                and self.last_hedge_price <= self.price_range_upper
                and (
                    current_price < self.price_range_lower
                    or current_price > self.price_range_upper
                )
            )
        )

        if price_change_pct >= self.price_threshold or price_crossed_boundary:
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
                    "crossed_boundary": price_crossed_boundary,
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
                # hedge_pnl = previous_hedge_position * (previous_price - current_price)
                hedge_pnl = previous_hedge_position * (current_price - previous_price)

                # 更新累积对冲盈亏
                self.cumulative_hedge_pnl += hedge_pnl

            # 计算无常损失
            il = self.calculate_impermanent_loss(current_price)

            # 价格是否在区间内
            in_range = self.price_range_lower <= current_price <= self.price_range_upper

            # 计算总价值
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
                    "in_range": in_range,
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

        # 计算价格在区间内的时间百分比
        in_range_pct = results["in_range"].mean() * 100

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
        final_gain_annual = hedge_vs_hodl_annual - impermanent_loss_annual

        # 计算资费和对冲成本的年化值
        funding_cost_pct = (self.total_funding_cost / self.initial_capital) * 100
        funding_cost_annual = annualize(funding_cost_pct)

        hedge_cost_pct = (self.total_hedge_cost / self.initial_capital) * 100
        hedge_cost_annual = annualize(hedge_cost_pct)

        # 计算总成本的年化值
        total_cost_pct = funding_cost_pct + hedge_cost_pct
        total_cost_annual = annualize(total_cost_pct)

        # 计算HODL盈亏和LP盈亏（绝对值，USDT）
        hodl_pnl = last_row["hodl_value"] - self.initial_capital
        lp_pnl = last_row["lp_value"] - self.initial_capital

        # 计算HODL收益率
        hodl_return = (last_row["hodl_value"] / self.initial_capital - 1) * 100
        hodl_return_annual = annualize(hodl_return)

        # 按照用户要求的格式返回摘要
        summary = {
            "## 初始参数：": "",
            "初始资金": self.initial_capital,
            "价格区间": f"[{self.price_range_lower}, {self.price_range_upper}]",
            "对冲区间": f"{self.price_threshold}",
            "价格在区间内的时间百分比": f"{in_range_pct:.2f}%",
            "时间段": f"{first_row['timestamp']} 至 {last_row['timestamp']}",
            "## 绝对表现": "",
            "对冲调整次数": self.hedge_count,
            "资金费用": f"{self.total_funding_cost:.2f} USDT",
            "对冲成本": f"{self.total_hedge_cost:.2f} USDT",
            "HODL盈亏": f"{hodl_pnl:.2f} USDT",
            "LP盈亏": f"{lp_pnl:.2f} USDT",
            "## 相对表现": "",
            "ETH价格变化": f"{(last_row['price'] / first_row['price'] - 1) * 100:.2f}%",
            "HODL收益率": f"{hodl_return:.2f}%",
            "不对冲收益率": f"{last_row['unhedged_return']:.2f}%",
            "对冲收益率": f"{last_row['hedged_return']:.2f}%",
            "对冲相对HODL收益率": f"{hedge_vs_hodl_return:.2f}%",
            "最终无常损失": f"{last_row['impermanent_loss']:.2f}%",
            "最终增益": f"{final_gain:.2f}%",
            "## 年化表现": "",
            "HODL收益率(年化)": f"{hodl_return_annual:.2f}%/年",
            "对冲相对HODL收益率(年化)": f"{hedge_vs_hodl_annual:.2f}%/年",
            "最终无常损失(年化)": f"{impermanent_loss_annual:.2f}%/年",
            "最终增益(年化)": f"{final_gain_annual:.2f}%/年",
            "对冲成本(年化)": f"{hedge_cost_annual:.2f}%/年",
            "资费(年化)": f"{funding_cost_annual:.2f}%/年",
        }

        return summary

    def plot_results(self, results, save_path=None):
        """绘制回测结果图表"""
        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 转换时间戳为datetime格式(如果是字符串)
        if isinstance(results["timestamp"].iloc[0], str):
            results["timestamp"] = pd.to_datetime(results["timestamp"])

        plt.figure(figsize=(15, 20))  # 增加高度以容纳五个子图

        # 图1: 价格与价格区间
        plt.subplot(5, 1, 1)
        plt.plot(results["timestamp"], results["price"], label="ETH价格", color="black")

        # 添加价格区间
        plt.axhline(
            y=self.price_range_lower, color="r", linestyle="--", label="区间下限"
        )
        plt.axhline(
            y=self.price_range_upper, color="g", linestyle="--", label="区间上限"
        )

        # 标记对冲点
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

            # 创建标签列表以避免重复
            labels_used = set()

            for hedge in sorted_history:
                hedge_time = (
                    pd.to_datetime(hedge["timestamp"])
                    if isinstance(hedge["timestamp"], str)
                    else hedge["timestamp"]
                )

                # 区分普通对冲和边界触发对冲
                if hedge.get("crossed_boundary", False):
                    label = "边界触发对冲" if "边界触发对冲" not in labels_used else ""
                    plt.axvline(
                        x=hedge_time, color="red", linestyle="-", alpha=0.3, label=label
                    )
                    if label:
                        labels_used.add(label)
                else:
                    label = "阈值触发对冲" if "阈值触发对冲" not in labels_used else ""
                    plt.axvline(
                        x=hedge_time,
                        color="blue",
                        linestyle="-",
                        alpha=0.3,
                        label=label,
                    )
                    if label:
                        labels_used.add(label)

        plt.title(
            f"ETH价格和价格区间 [{self.price_range_lower}, {self.price_range_upper}]"
        )
        plt.ylabel("价格 (USDT)")
        plt.legend()
        plt.grid(True)

        # 图2: 收益率对比
        plt.subplot(5, 1, 2)
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
        plt.subplot(5, 1, 3)
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

        # 图4: 资产分布
        plt.subplot(5, 1, 4)
        plt.stackplot(
            results["timestamp"],
            [results["eth_amount"] * results["price"], results["usdt_amount"]],
            labels=["ETH价值", "USDT价值"],
            colors=["orange", "green"],
            alpha=0.5,
        )
        plt.title("LP资产分布")
        plt.ylabel("价值 (USDT)")
        plt.legend(loc="upper left")
        plt.grid(True)

        # 图5: 成本
        plt.subplot(5, 1, 5)
        plt.plot(results["timestamp"], results["total_funding_cost"], label="资金费用")
        plt.plot(results["timestamp"], results["hedge_cost"], label="对冲成本")
        plt.title("对冲相关成本")
        plt.ylabel("USDT")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 如果指定了保存路径，则保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path:
            plt.savefig(
                f"{save_path}/v3_backtest_results_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
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
                f"{save_dir}/v3_hedge_history_{timestamp}.csv", index=False
            )

        # 保存回测结果
        results.to_csv(f"{save_dir}/v3_backtest_results_{timestamp}.csv", index=False)

        # 保存图表
        self.plot_results(results, save_path=save_dir)

        # 保存摘要
        summary = self.get_summary(results)
        with open(f"{save_dir}/v3_summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            current_section = ""
            for key, value in summary.items():
                # 处理章节标题
                if key.startswith("##"):
                    current_section = key
                    f.write(f"{key}\n\n")
                else:
                    f.write(f"{key}: {value}\n")

                # 在每个章节结束时添加空行
                if (
                    key == "时间段"
                    or key == "LP盈亏"
                    or key == "最终增益"
                    or key == "资费(年化)"
                ):
                    f.write("\n")

        print(f"结果已保存到目录: {save_dir}")

    def plot_range_comparison(self, results_dict, save_path=None):
        """绘制不同价格区间的对比图"""
        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        plt.figure(figsize=(10, 14))

        # 获取第一个结果的数据用于基准
        first_key = list(results_dict.keys())[0]
        first_results = results_dict[first_key]["results"]

        # 图1: 价格和不同区间
        plt.subplot(4, 1, 1)
        plt.plot(
            first_results["timestamp"],
            first_results["price"],
            label="ETH价格",
            color="black",
        )

        # 为每个区间绘制范围线
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))
        for i, ((range_lower, range_upper), results_data) in enumerate(
            results_dict.items()
        ):
            color = colors[i]
            label = f"区间 [{range_lower}, {range_upper}]"
            plt.axhline(
                y=range_lower,
                color=color,
                linestyle="--",
                alpha=0.5,
                label=f"{label} 下限",
            )
            plt.axhline(
                y=range_upper,
                color=color,
                linestyle="-.",
                alpha=0.5,
                label=f"{label} 上限",
            )

        plt.title("ETH价格和不同价格区间")
        plt.ylabel("价格 (USDT)")
        plt.legend()
        plt.grid(True)

        # 图2: 不同区间的收益率对比
        plt.subplot(4, 1, 2)
        for i, ((range_lower, range_upper), results_data) in enumerate(
            results_dict.items()
        ):
            label = f"区间 [{range_lower}, {range_upper}]"
            plt.plot(
                results_data["results"]["timestamp"],
                results_data["results"]["hedged_return"],
                label=f"{label} 对冲收益率",
                color=colors[i],
            )

        # 添加HODL收益率作为基准
        hodl_return = (first_results["hodl_value"] / self.initial_capital - 1) * 100
        plt.plot(
            first_results["timestamp"],
            hodl_return,
            label="HODL收益率",
            color="black",
            linestyle=":",
        )

        plt.title("不同价格区间的对冲收益率对比")
        plt.ylabel("收益率 (%)")
        plt.legend()
        plt.grid(True)

        # 图3: 不同区间的无常损失对比
        plt.subplot(4, 1, 3)
        for i, ((range_lower, range_upper), results_data) in enumerate(
            results_dict.items()
        ):
            label = f"区间 [{range_lower}, {range_upper}]"
            plt.plot(
                results_data["results"]["timestamp"],
                results_data["results"]["impermanent_loss"],
                label=f"{label} 无常损失",
                color=colors[i],
            )

        plt.title("不同价格区间的无常损失对比")
        plt.ylabel("无常损失 (%)")
        plt.legend()
        plt.grid(True)

        # 图4: 区间内时间百分比
        plt.subplot(4, 1, 4)
        ranges = []
        in_range_pcts = []
        for (range_lower, range_upper), results_data in results_dict.items():
            ranges.append(f"[{range_lower}, {range_upper}]")
            in_range_pcts.append(results_data["results"]["in_range"].mean() * 100)

        plt.bar(ranges, in_range_pcts, color=colors)
        plt.title("价格在各区间内的时间百分比")
        plt.ylabel("时间百分比 (%)")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")

        plt.tight_layout()

        # 如果指定了保存路径，则保存图表
        if save_path:
            plt.savefig(
                f"{save_path}/v3_range_comparison.png", dpi=300, bbox_inches="tight"
            )

        # plt.show()


# 使用示例
if __name__ == "__main__":
    import pandas as pd

    # 初始化参数
    initial_capital = 1000  # 10万USDT
    price_threshold = 0.05  # 5%价格波动触发对冲
    funding_rate = 0.1  # 10%年化资金费率

    # 创建保存目录
    save_dir = "v3_backtest_results"
    os.makedirs(save_dir, exist_ok=True)

    # 读取真实数据
    # data = pd.read_csv("data/ETH_BTC_calculated_15min_cop.csv")
    # filename = "ETH_BTC_calculated_15min_41.csv"
    # filename = "ETH_BTC_calculated_15min_cop.csv"
    # filename = "ETH_USDT_1h.csv"
    # filename = "ETH_BTC_calculated_1h.csv"
    filename = "APT_USDT_1h.csv"
    data = pd.read_csv(f"data/{filename}")
    print(
        f"加载数据: {len(data)} 条记录，时间范围: {data['timestamp'].iloc[0]} 至 {data['timestamp'].iloc[-1]}"
    )

    # 数据必须包含timestamp和close字段
    if "timestamp" not in data.columns or "close" not in data.columns:
        raise ValueError("数据文件必须包含timestamp和close字段")

    # 获取初始价格和价格区间
    initial_price = data.iloc[0]["close"]
    print(f"初始价格: {initial_price}")

    # 设置价格区间为初始价格的±10%
    price_range_upper = initial_price * 20
    price_range_lower = initial_price / (price_range_upper / initial_price)
    print(f"设置价格区间: [{price_range_lower:.2f}, {price_range_upper:.2f}]")

    # 创建策略实例
    strategy = UniswapV3Strategy(
        initial_capital=initial_capital,
        price_range_lower=price_range_lower,
        price_range_upper=price_range_upper,
        price_threshold=price_threshold,
        funding_rate=funding_rate,
        gas_cost_per_hedge=0.0003,  # 对冲成本为交易量的0.05%
    )

    try:
        # 运行回测
        results = strategy.run(data)

        # 保存结果
        strategy.save_results(results, save_dir)

        # 打印摘要
        summary = strategy.get_summary(results)
        print("\nUniswap V3 回测结果摘要:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
