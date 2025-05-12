import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Visualizer:
    """回测结果可视化工具"""

    def __init__(self, output_dir="data/results"):
        """
        初始化可视化工具

        Args:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置Seaborn风格
        sns.set_style("whitegrid")
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

    def plot_price_chart(self, df, title, filename):
        """
        绘制价格走势图

        Args:
            df (pd.DataFrame): 价格数据，包含'timestamp'和'close'列
            title (str): 图表标题
            filename (str): 输出文件名
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["close"], linewidth=1)
        plt.title(f"{title}价格走势图")
        plt.xlabel("时间")
        plt.ylabel("价格")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_price.png"), dpi=300)
        plt.close()

    def plot_relative_price(self, df, base, quote, filename):
        """
        绘制相对价格走势图

        Args:
            df (pd.DataFrame): 相对价格数据，包含'timestamp'和'relative_close'列
            base (str): 基础币种，如 ETH
            quote (str): 报价币种，如 BTC
            filename (str): 输出文件名
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["relative_close"], linewidth=1)
        plt.title(f"{base}/{quote}相对价格走势图")
        plt.xlabel("时间")
        plt.ylabel(f"{base}/{quote}价格比")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_relative.png"), dpi=300)
        plt.close()

    def plot_backtest_results(self, results, strategy_name, params=None):
        """
        绘制回测结果

        Args:
            results (pd.DataFrame): 回测结果
            strategy_name (str): 策略名称
            params (dict, optional): 策略参数
        """
        # 创建文件名
        if params:
            param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
            filename = f"{strategy_name}_{param_str}"
        else:
            filename = strategy_name

        # 绘制多子图
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 15), gridspec_kw={"height_ratios": [2, 1, 1]}
        )

        # 1. 价格和持仓
        ax1 = axes[0]
        ax1.plot(results["timestamp"], results["price"], label="价格", color="blue")
        ax1.set_ylabel("价格", color="blue")
        ax1.set_title(f"{strategy_name}回测结果")

        # 添加持仓信息到第二个y轴
        if "position" in results.columns:
            ax1_twin = ax1.twinx()
            ax1_twin.fill_between(
                results["timestamp"], results["position"], 0, alpha=0.3, color="green"
            )
            ax1_twin.set_ylabel("持仓量", color="green")

        # 2. 累计收益
        ax2 = axes[1]
        ax2.plot(
            results["timestamp"], results["total_pnl"], label="累计收益", color="orange"
        )
        ax2.hlines(
            y=0,
            xmin=results["timestamp"].iloc[0],
            xmax=results["timestamp"].iloc[-1],
            colors="black",
            linestyles="dashed",
            alpha=0.7,
        )
        ax2.set_ylabel("累计收益")

        # 3. 每日收益率
        if "daily_return" in results.columns:
            ax3 = axes[2]
            ax3.bar(results["timestamp"], results["daily_return"], width=0.7)
            ax3.set_ylabel("每日收益率")
        else:
            # 如果没有每日收益率数据，可以绘制其他相关指标，如波动性
            ax3 = axes[2]
            # 计算滚动波动率
            if len(results) > 20:  # 确保有足够数据点计算波动率
                volatility = results["price"].pct_change().rolling(20).std() * np.sqrt(
                    365
                )
                ax3.plot(
                    results["timestamp"], volatility, label="波动率(年化)", color="red"
                )
                ax3.set_ylabel("波动率")
            else:
                ax3.set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_results.png"), dpi=300)
        plt.close()

    def plot_sensitivity_analysis(self, results_df, x_param, y_param, z_param, title):
        """
        绘制敏感性分析热图

        Args:
            results_df (pd.DataFrame): 包含不同参数组合回测结果的DataFrame
            x_param (str): x轴参数名称
            y_param (str): y轴参数名称
            z_param (str): 热图颜色代表的参数(如收益率)
            title (str): 图表标题
        """
        # 数据透视
        pivot_table = results_df.pivot_table(
            index=y_param, columns=x_param, values=z_param
        )

        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
        plt.title(f"{title}敏感性分析")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir, f"sensitivity_{x_param}_{y_param}_{z_param}.png"
            ),
            dpi=300,
        )
        plt.close()

    def plot_amm_comparison(self, results_dict, metric="total_pnl"):
        """
        比较不同AMM曲线的性能

        Args:
            results_dict (dict): 字典，键为AMM名称，值为结果DataFrame
            metric (str): 要比较的指标
        """
        plt.figure(figsize=(12, 6))

        for amm_name, results in results_dict.items():
            plt.plot(results["timestamp"], results[metric], label=amm_name)

        plt.title(f"不同AMM曲线{metric}对比")
        plt.xlabel("时间")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"amm_comparison_{metric}.png"), dpi=300
        )
        plt.close()


if __name__ == "__main__":
    # 使用示例
    visualizer = Visualizer()

    # 创建模拟数据
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = {
        "timestamp": dates,
        "price": np.cumsum(np.random.normal(0, 1, 100)) + 100,
        "position": np.random.randint(0, 10, 100),
        "total_pnl": np.cumsum(np.random.normal(0.001, 0.01, 100)),
    }
    df = pd.DataFrame(data)

    # 绘制图表
    visualizer.plot_backtest_results(df, "测试策略", {"param1": 0.1, "param2": 0.2})
