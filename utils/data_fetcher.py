import ccxt
import pandas as pd
from datetime import datetime
import os
import time
from requests.exceptions import RequestException
import logging
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        # 配置代理设置
        proxies = {
            "http": "http://127.0.0.1:7890",  # 根据您的代理设置修改
            "https": "http://127.0.0.1:7890",  # 根据您的代理设置修改
        }

        # 创建会话
        session = requests.Session()
        session.proxies.update(proxies)

        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "timeout": 30000,  # 30秒超时
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,
                },
                "session": session,
                "proxies": proxies,
            }
        )
        self.max_retries = 3
        self.retry_delay = 5  # 重试延迟秒数

    def fetch_ohlcv(self, symbol, start_date, end_date, timeframe="1h"):
        """
        获取历史K线数据，包含重试机制

        Args:
            symbol (str): 交易对，例如 'ETH/USDT'
            start_date (str): 开始日期，格式 'YYYY-MM-DD'
            end_date (str): 结束日期，格式 'YYYY-MM-DD'
            timeframe (str): 时间周期，默认 '1h'

        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame
        """
        # 转换日期格式
        start_timestamp = int(
            datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000
        )
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        for attempt in range(self.max_retries):
            try:
                # 获取数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=start_timestamp, limit=100000
                )

                # 转换为DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                # 转换时间戳
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                return df

            except (ccxt.NetworkError, ccxt.ExchangeError, RequestException) as e:
                logger.warning(f"尝试 {attempt + 1}/{self.max_retries} 失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"获取数据失败，已达到最大重试次数: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"发生未知错误: {str(e)}")
                raise

    def save_data(self, df, symbol, timeframe):
        """
        保存数据到本地

        Args:
            df (pd.DataFrame): 要保存的数据
            symbol (str): 交易对
            timeframe (str): 时间周期
        """
        try:
            # 创建数据目录
            os.makedirs("data", exist_ok=True)

            # 生成文件名
            filename = f'data/{symbol.replace("/", "_")}_{timeframe}.csv'

            # 保存数据
            df.to_csv(filename, index=False)
            logger.info(f"数据已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存数据时发生错误: {str(e)}")
            raise

    def load_data(self, symbol, timeframe):
        """
        从本地加载数据

        Args:
            symbol (str): 交易对
            timeframe (str): 时间周期

        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            filename = f'data/{symbol.replace("/", "_")}_{timeframe}.csv'

            if os.path.exists(filename):
                df = pd.read_csv(filename, parse_dates=["timestamp"])
                logger.info(f"从本地加载数据: {filename}")
                return df
            return None
        except Exception as e:
            logger.error(f"加载数据时发生错误: {str(e)}")
            return None
