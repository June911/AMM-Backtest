import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from backtest.uniswap_v3_strategy import UniswapV3Strategy


def analyze_thresholds(data, thresholds, save_dir=None):
    """
    分析不同price_threshold值对最终增益的影响

    Args:
        data (pd.DataFrame): 价格数据，包含timestamp和close字段
        thresholds (list): 要测试的price_threshold值列表
        save_dir (str, optional): 结果保存目录，默认为None

    Returns:
        pd.DataFrame: 包含不同阈值的分析结果
    """
    # 获取初始价格
    initial_price = data.iloc[0]["close"]

    # 设置共同的参数
    initial_capital = 100000  # 10万USDT
    funding_rate = 0.01  # 1%年化资金费率
    gas_cost_per_hedge = 0.0004  # 对冲成本为交易量的0.05%

    # 设置价格区间为初始价格的±x%
    price_ratio = 10
    price_range_upper = initial_price * price_ratio
    price_range_lower = initial_price / (price_range_upper / initial_price)

    # 存储每个阈值的结果
    threshold_results = {}
    threshold_metrics = []

    print(f"初始价格: {initial_price}")
    print(f"价格区间: [{price_range_lower}, {price_range_upper}]")

    # 为每个阈值运行回测
    for threshold in thresholds:
        print(f"\n测试price_threshold={threshold}...")

        # 创建策略实例
        strategy = UniswapV3Strategy(
            initial_capital=initial_capital,
            price_range_lower=price_range_lower,
            price_range_upper=price_range_upper,
            price_threshold=threshold,
            funding_rate=funding_rate,
            gas_cost_per_hedge=gas_cost_per_hedge,
        )

        # 运行回测
        results = strategy.run(data)

        # 保存结果
        threshold_results[threshold] = {
            "strategy": strategy,
            "results": results,
        }

        # 获取摘要
        summary = strategy.get_summary(results)

        # 保存关键指标
        threshold_metrics.append(
            {
                "price_threshold": threshold,
                "hedge_count": summary["对冲调整次数"],
                "hedged_return": float(summary["对冲收益率"].strip("%")),
                "unhedged_return": float(summary["不对冲收益率"].strip("%")),
                "impermanent_loss": float(summary["最终无常损失"].strip("%")),
                "final_gain": float(summary["最终增益"].strip("%")),
                "hedge_cost": float(summary["对冲成本"].split()[0]),
                "funding_cost": float(summary["资金费用"].split()[0]),
                "total_cost": float(summary["对冲成本"].split()[0])
                + float(summary["资金费用"].split()[0]),
                "price_change_pct": float(summary.get("ETH价格变化", "0").strip("%")),
                "hodl_pnl": float(summary.get("HODL盈亏", "0").split()[0]),
                "lp_pnl": float(summary.get("LP盈亏", "0").split()[0]),
                "hedged_vs_hodl": float(
                    summary.get("对冲相对HODL收益率", "0").strip("%")
                ),
                "in_range_time_pct": float(
                    summary.get("价格在区间内的时间百分比", "0").strip("%")
                ),
                "start_time": summary.get("开始时间", data["timestamp"].iloc[0]),
                "end_time": summary.get("结束时间", data["timestamp"].iloc[-1]),
                # 年化指标
                "annual_hedged_vs_hodl": float(
                    summary.get("对冲相对HODL收益率(年化)", "0").strip("%/年")
                ),
                "annual_impermanent_loss": float(
                    summary.get("最终无常损失(年化)", "0").strip("%/年")
                ),
                "annual_final_gain": float(
                    summary.get("最终增益(年化)", "0").strip("%/年")
                ),
                "annual_hedge_cost": float(
                    summary.get("对冲成本(年化)", "0").strip("%/年")
                ),
                "annual_funding_cost": float(
                    summary.get("资费(年化)", "0").strip("%/年")
                ),
            }
        )

        print(f"  对冲调整次数: {summary['对冲调整次数']}")
        print(f"  最终增益: {summary['最终增益']}")

    # 创建结果DataFrame
    metrics_df = pd.DataFrame(threshold_metrics)

    # 如果指定了保存目录，则保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 保存指标
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_df.to_csv(f"{save_dir}/threshold_analysis_{timestamp}.csv", index=False)

        # 生成一个更详细的结果CSV
        detailed_results = []
        for threshold in thresholds:
            strategy = threshold_results[threshold]["strategy"]
            summary = strategy.get_summary(threshold_results[threshold]["results"])

            detailed_results.append(
                {
                    "价格波动阈值": threshold,
                    # 初始参数
                    "初始资金": initial_capital,
                    "价格区间下限": price_range_lower,
                    "价格区间上限": price_range_upper,
                    "对冲区间": threshold,
                    "价格在区间内的时间百分比": summary.get(
                        "价格在区间内的时间百分比", "100.00%"
                    ),
                    "开始时间": summary.get("开始时间", ""),
                    "结束时间": summary.get("结束时间", ""),
                    # 绝对表现
                    "对冲调整次数": summary["对冲调整次数"],
                    "资金费用": summary["资金费用"],
                    "对冲成本": summary["对冲成本"],
                    "HODL盈亏": summary.get("HODL盈亏", "N/A"),
                    "LP盈亏": summary.get("LP盈亏", "N/A"),
                    # 相对表现
                    "ETH价格变化": summary.get("ETH价格变化", "N/A"),
                    "不对冲收益率": summary["不对冲收益率"],
                    "对冲收益率": summary.get("对冲收益率", "N/A"),
                    "对冲相对HODL收益率": summary.get("对冲相对HODL收益率", "N/A"),
                    "最终无常损失": summary["最终无常损失"],
                    "最终增益": summary["最终增益"],
                    # 年化表现
                    "对冲相对HODL收益率(年化)": summary.get(
                        "对冲相对HODL收益率(年化)", "N/A"
                    ),
                    "最终无常损失(年化)": summary.get("最终无常损失(年化)", "N/A"),
                    "最终增益(年化)": summary.get("最终增益(年化)", "N/A"),
                    "对冲成本(年化)": summary.get("对冲成本(年化)", "N/A"),
                    "资费(年化)": summary.get("资费(年化)", "N/A"),
                }
            )

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(
            f"{save_dir}/threshold_detailed_results_{timestamp}.csv", index=False
        )

        print(
            f"\n详细结果已保存至: {save_dir}/threshold_detailed_results_{timestamp}.csv"
        )

        # 生成并保存可视化图表
        plot_threshold_analysis(metrics_df, threshold_results, save_dir)

    return metrics_df, threshold_results


def plot_threshold_analysis(metrics_df, threshold_results, save_dir):
    """生成阈值分析的可视化图表"""
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建时间戳用于保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 图1：阈值与最终增益的关系
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df["price_threshold"], metrics_df["final_gain"], "o-", linewidth=2)
    plt.xlabel("价格波动阈值")
    plt.ylabel("最终增益 (%)")
    plt.title("价格波动阈值与最终增益的关系")
    plt.grid(True)
    plt.savefig(
        f"{save_dir}/threshold_vs_final_gain_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 图2：阈值与对冲次数的关系
    plt.figure(figsize=(12, 6))
    plt.plot(
        metrics_df["price_threshold"],
        metrics_df["hedge_count"],
        "o-",
        linewidth=2,
        color="orange",
    )
    plt.xlabel("价格波动阈值")
    plt.ylabel("对冲调整次数")
    plt.title("价格波动阈值与对冲调整次数的关系")
    plt.grid(True)
    plt.savefig(
        f"{save_dir}/threshold_vs_hedge_count_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 图3：阈值、对冲次数和最终增益的组合图
    fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()

    # 左轴：最终增益
    ax1.plot(
        metrics_df["price_threshold"],
        metrics_df["final_gain"],
        "o-",
        color="blue",
        linewidth=2,
        label="最终增益 (%)",
    )
    ax1.set_xlabel("价格波动阈值")
    ax1.set_ylabel("最终增益 (%)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # 右轴：对冲次数
    ax2 = ax1.twinx()
    ax2.plot(
        metrics_df["price_threshold"],
        metrics_df["hedge_count"],
        "o-",
        color="orange",
        linewidth=2,
        label="对冲调整次数",
    )
    ax2.set_ylabel("对冲调整次数", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    plt.title("价格波动阈值、最终增益与对冲调整次数的关系")
    plt.grid(True, alpha=0.3)

    # 添加两个y轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/threshold_gain_hedgecount_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 图4：阈值与成本的关系
    plt.figure(figsize=(12, 6))
    plt.plot(
        metrics_df["price_threshold"],
        metrics_df["hedge_cost"],
        "o-",
        label="对冲成本",
        linewidth=2,
    )
    plt.plot(
        metrics_df["price_threshold"],
        metrics_df["funding_cost"],
        "o-",
        label="资金费用",
        linewidth=2,
    )
    plt.plot(
        metrics_df["price_threshold"],
        metrics_df["total_cost"],
        "o-",
        label="总成本",
        linewidth=2,
    )
    plt.xlabel("价格波动阈值")
    plt.ylabel("成本 (USDT)")
    plt.title("价格波动阈值与各类成本的关系")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        f"{save_dir}/threshold_vs_costs_{timestamp}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 图5：最终增益与对冲次数的散点图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        metrics_df["hedge_count"],
        metrics_df["final_gain"],
        c=metrics_df["price_threshold"],
        s=100,
        cmap="viridis",
        alpha=0.8,
    )

    # 为每个点添加阈值标签
    for i, threshold in enumerate(metrics_df["price_threshold"]):
        plt.annotate(
            f"{threshold}",
            (metrics_df["hedge_count"].iloc[i], metrics_df["final_gain"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.colorbar(scatter, label="价格波动阈值")
    plt.xlabel("对冲调整次数")
    plt.ylabel("最终增益 (%)")
    plt.title("对冲调整次数与最终增益的关系")
    plt.grid(True)
    plt.savefig(
        f"{save_dir}/hedge_count_vs_final_gain_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 图6：比较不同阈值的收益率曲线
    plt.figure(figsize=(15, 8))

    # 获取第一个结果的时间戳数据
    first_key = list(threshold_results.keys())[0]
    first_results = threshold_results[first_key]["results"]

    # 绘制每个阈值的对冲收益率
    colors = plt.cm.viridis(np.linspace(0, 1, len(threshold_results)))
    for i, (threshold, results_data) in enumerate(threshold_results.items()):
        plt.plot(
            results_data["results"]["timestamp"],
            results_data["results"]["hedged_return"],
            label=f"阈值={threshold}",
            color=colors[i],
            linewidth=1.5,
        )

    # 添加HODL基准
    hodl_return = (first_results["hodl_value"] / 100000 - 1) * 100
    plt.plot(
        first_results["timestamp"],
        hodl_return,
        label="HODL策略",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    plt.title("不同价格波动阈值的对冲收益率对比")
    plt.xlabel("时间")
    plt.ylabel("收益率 (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"{save_dir}/hedged_returns_comparison_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    # 读取真实数据
    # filename = "ETH_BTC_calculated_15min_41.csv"
    # filename = "ETH_BTC_calculated_15min_cop.csv"
    # filename = "ETH_USDT_1h.csv"
    # filename = "ETH_BTC_calculated_1h.csv"
    filename = "APT_USDT_1h.csv"
    data = pd.read_csv(f"data/{filename}")
    print(
        f"加载数据: {len(data)} 条记录，时间范围: {data['timestamp'].iloc[0]} 至 {data['timestamp'].iloc[-1]}"
    )

    # 测试的阈值范围
    thresholds = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]

    # 创建保存目录
    save_dir = f"v3_threshold_analysis_{filename}"

    # 运行分析
    metrics_df, threshold_results = analyze_thresholds(data, thresholds, save_dir)

    # 打印分析结果
    print("\n阈值分析结果摘要:")
    print(
        metrics_df[["price_threshold", "hedge_count", "final_gain"]].to_string(
            index=False
        )
    )

    # 寻找最佳阈值
    best_gain_idx = metrics_df["final_gain"].idxmax()
    best_threshold = metrics_df.loc[best_gain_idx, "price_threshold"]
    best_gain = metrics_df.loc[best_gain_idx, "final_gain"]

    print(f"\n最佳阈值: {best_threshold} (最终增益: {best_gain:.2f}%)")
