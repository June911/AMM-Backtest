import os
import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta


class DataLoader:
    """数据加载和处理工具"""

    def __init__(self, data_dir="data"):
        """
        初始化数据加载器

        Args:
            data_dir (str): 数据存储目录
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")

        # 确保目录存在
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # 初始化交易所API
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,  # 启用请求速率限制
            }
        )

    def fetch_ohlcv(self, symbol, start_date, end_date, timeframe="1h"):
        """
        获取OHLCV数据

        Args:
            symbol (str): 交易对
            start_date (str): 开始日期，格式：YYYY-MM-DD
            end_date (str): 结束日期，格式：YYYY-MM-DD
            timeframe (str): 时间周期

        Returns:
            pd.DataFrame: OHLCV数据
        """
        filename = f"{self.raw_dir}/{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.csv"

        # 检查是否已存在缓存数据
        if os.path.exists(filename):
            print(f"加载已缓存的数据: {filename}")
            return pd.read_csv(filename, parse_dates=["timestamp"])

        print(f"从交易所获取新数据: {symbol} {start_date} 到 {end_date}")

        # 转换日期字符串为时间戳
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_candles = []

        while since < end_timestamp:
            try:
                # 获取K线数据
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since, limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # 更新since时间戳，为最后一根K线的时间戳加一
                since = candles[-1][0] + 1

                # API速率限制
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"获取数据出错: {e}")
                # 出错后等待一段时间再重试
                time.sleep(10)

        # 转换为DataFrame
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # 转换时间戳为datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # 保存数据到文件
        df.to_csv(filename, index=False)

        return df

    def process_relative_price(
        self, base_symbol, quote_symbol, start_date, end_date, timeframe="1h"
    ):
        """
        处理相对价格(base/quote)

        Args:
            base_symbol (str): 基础币种，如 ETH/USDT
            quote_symbol (str): 报价币种，如 BTC/USDT
            start_date (str): 开始日期
            end_date (str): 结束日期
            timeframe (str): 时间周期

        Returns:
            pd.DataFrame: 处理后的相对价格数据
        """
        # 获取基础币种和报价币种的价格
        base_data = self.fetch_ohlcv(base_symbol, start_date, end_date, timeframe)
        quote_data = self.fetch_ohlcv(quote_symbol, start_date, end_date, timeframe)

        # 确保时间戳对齐
        base_data = base_data.set_index("timestamp")
        quote_data = quote_data.set_index("timestamp")

        # 取交集
        common_index = base_data.index.intersection(quote_data.index)

        if len(common_index) == 0:
            raise ValueError("两个数据集没有共同的时间戳")

        base_aligned = base_data.loc[common_index]
        quote_aligned = quote_data.loc[common_index]

        # 计算相对价格
        relative_df = pd.DataFrame(index=common_index)
        relative_df["timestamp"] = common_index
        relative_df["relative_open"] = base_aligned["open"] / quote_aligned["open"]
        relative_df["relative_high"] = base_aligned["high"] / quote_aligned["high"]
        relative_df["relative_low"] = base_aligned["low"] / quote_aligned["low"]
        relative_df["relative_close"] = base_aligned["close"] / quote_aligned["close"]
        relative_df["base_volume"] = base_aligned["volume"]
        relative_df["quote_volume"] = quote_aligned["volume"]

        # 重置索引
        relative_df = relative_df.reset_index(drop=True)

        # 保存处理后的数据
        output_file = f"{self.processed_dir}/{base_symbol.split('/')[0]}_{quote_symbol.split('/')[0]}_{timeframe}.csv"
        relative_df.to_csv(output_file, index=False)

        return relative_df

    def load_processed_data(self, base, quote, timeframe="1h"):
        """
        加载处理后的数据

        Args:
            base (str): 基础币种，如 ETH
            quote (str): 报价币种，如 BTC
            timeframe (str): 时间周期

        Returns:
            pd.DataFrame: 处理后的数据
        """
        filename = f"{self.processed_dir}/{base}_{quote}_{timeframe}.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到处理后的数据文件: {filename}")

        return pd.read_csv(filename, parse_dates=["timestamp"])


if __name__ == "__main__":
    # 使用示例
    loader = DataLoader()

    # 获取ETH/USDT和BTC/USDT的数据，计算相对价格ETH/BTC
    try:
        relative_prices = loader.process_relative_price(
            "ETH/USDT",
            "BTC/USDT",
            start_date="2025-01-01",
            end_date="2025-03-01",
            timeframe="1h",
        )
        print(f"计算了 {len(relative_prices)} 条相对价格数据")
        print(relative_prices.head())
    except Exception as e:
        print(f"处理数据时出错: {e}")
