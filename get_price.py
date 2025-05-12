import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher

matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Arial Unicode MS",
]  # 任选安装好的中文字体
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def plot_results(results, title):
    """
    绘制回测结果

    Args:
        results (pd.DataFrame): 回测结果
        title (str): 图表标题
    """
    # ---------- 新增: 确保输出目录存在，清理文件名 ----------
    output_dir = os.path.join("data", "results")
    os.makedirs(output_dir, exist_ok=True)
    # 将文件名中的 / 等非法字符替换为下划线
    safe_title = title.replace("/", "_").replace(" ", "_")
    # -----------------------------------------------------------

    plt.figure(figsize=(15, 10))

    # 价格变化
    plt.plot(results["timestamp"], results["close"])
    plt.title("价格走势")
    plt.xlabel("时间")
    plt.ylabel("价格")

    plt.tight_layout()
    # ----------- 修改保存路径 -------------
    plt.savefig(os.path.join(output_dir, f"{safe_title}.png"))
    plt.close()


# ---------------- 新增: 封装抓取逻辑为函数 -----------------
def fetch_price_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1h",
    force_refresh: bool = False,
):
    """按需抓取指定交易对历史数据

    Args:
        symbol (str): 交易对，如 "APT/USDT"
        start_date (str): 起始日期 YYYY-MM-DD
        end_date (str): 结束日期 YYYY-MM-DD
        timeframe (str, optional): K 线周期. Defaults to "1h".
        force_refresh (bool, optional): 是否忽略已有文件重新抓取. Defaults to False.

    Returns:
        pd.DataFrame: 返回包含 timestamp 与 OHLCV 的 DataFrame
    """

    # 将文件名规范化，便于判断是否已有数据
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join("data", filename)

    # 如果本地已存在数据且不强制刷新，直接读取
    if os.path.exists(filepath) and not force_refresh:
        print(f"已检测到本地数据文件 {filepath}，直接读取")
        return pd.read_csv(filepath)

    print("开始抓取历史数据……")

    fetcher = DataFetcher()

    all_data = []
    current_start = start_date

    while True:
        batch_data = fetcher.fetch_ohlcv(symbol, current_start, end_date, timeframe)

        if batch_data is None or len(batch_data) == 0:
            break

        all_data.append(batch_data)

        # 如果获取的数据量小于限制，说明已经获取完所有数据
        if len(batch_data) < 1000:
            break

        # 更新下一批数据的开始时间
        last_timestamp = batch_data["timestamp"].iloc[-1]
        current_start = last_timestamp.strftime("%Y-%m-%d")

        # 避免重复获取最后一条记录
        batch_data = batch_data.iloc[:-1]

    # 合并所有批次的数据
    if all_data:
        data = pd.concat(all_data, ignore_index=True)
        # 去除重复并排序
        data = data.drop_duplicates(subset=["timestamp"])
        data = data.sort_values("timestamp")

        # 保存完整数据
        fetcher.save_data(data, symbol, timeframe)

        return data

    raise ValueError("未能成功获取任何数据，请检查参数是否正确或网络是否可用")


# ------------------------------------------------------------


def main():
    # 参数设置
    symbol = "APT/USDT"
    start_date = "2025-04-01"
    end_date = "2025-04-10"
    timeframe = "1h"

    # 如果本地不存在对应数据，则自动抓取
    data = fetch_price_data(symbol, start_date, end_date, timeframe)

    # 绘制价格走势图
    plot_results(data, symbol)


if __name__ == "__main__":
    main()
