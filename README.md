# DEX LP 对冲回测框架

本项目用于回测不同 AMM 曲线下的 LP 策略，主要研究合约对冲对无常损失的影响。通过模拟真实市场环境，帮助用户评估不同对冲策略的效果。

## AMM 曲线支持

本项目支持以下 AMM 曲线：

### Balancer
- 支持自定义权重池
- Uniswap v2 作为 Balancer 的特殊形式（50/50 权重池）

### Uniswap v3
- 支持集中流动性
- 支持自定义价格区间

### Curve
- 支持稳定币交易对
- 支持自定义 A 参数

## 回测流程

### 1. 价格数据获取与处理
- 获取原始价格数据
- 价格处理：
  - 计算相对价格（如 ETH/BTC）
  - 数据清洗和标准化
- 存储处理后的价格数据

### 2. 回测执行
- 支持多种 AMM 曲线
- 支持合约对冲策略
- 计算关键指标：
  - 无常损失
  - LP 手续费收益（`fee_apr` 参数）
  - 对冲收益
  - 总收益率

### 3. 敏感性分析
- 参数敏感性测试
- 市场条件敏感性分析
- 对冲策略效果评估

## 项目结构
```
.
├── backtest/          # 回测相关代码
│   ├── main.py       # 主回测程序
│   └── strategy.py   # 策略实现
├── data/             # 数据存储目录
│   ├── raw/         # 原始数据
│   └── processed/   # 处理后的数据
├── utils/            # 工具函数
│   ├── data_loader.py    # 数据加载工具
│   └── visualization.py  # 可视化工具
├── requirements.txt  # 项目依赖
└── README.md         # 项目说明
```

## 功能特点
- 支持多种 AMM 曲线
- 小时级别数据回测
- 支持合约对冲策略
- 可视化回测结果
- 支持自定义策略参数
- 提供详细的回测报告

## 环境要求
- Python 3.8+
- pip 包管理器

## 安装步骤
1. 克隆项目：
```bash
git clone https://github.com/yourusername/uniswap-v2-lp-backtest.git
cd uniswap-v2-lp-backtest
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法
1. 准备数据：
```bash
python get_price.py
```

2. 运行回测：
```bash
python backtest/uniswap_v2_strategy.py
python backtest/uniswap_v3_strategy.py
```

3. 查看结果：
回测结果将保存在 `data/results` 目录下，包括：
- 回测报告（CSV格式）
- 收益曲线图
- 对冲效果分析
- 敏感性分析报告

## 参数配置
- 回测时间段：可配置
- 数据频率：小时级别
- AMM 参数：
  - 池子权重（Balancer）
  - 价格区间（Uniswap v3）
  - A 参数（Curve）
- 对冲参数：
  - 价格波动阈值
  - 对冲比例
  - 资金费率
- LP 手续费参数：
  - `fee_apr`：手续费年化率，默认 0（不计费，保持旧行为）
- 初始资金：可配置

## 手续费收益模块

LP 的收入端通过 `fee_apr` 参数建模：按当前 LP 仓位价值 × 行间实际时间差线性累积
（小时费率 = `fee_apr / 8760`；首行无时间流逝不计费，缺失 K 线按实际间隔补足时长），
累积到独立的 `total_fee_income`，**不复投**回池子，
因此无常损失始终是纯价格效应，收入/成本各自成行，口径干净。

- **V2**：全程计费。`fee_apr` 取目标池子的手续费 APR ≈ 日成交量 × 费率 / TVL × 365。
- **V3**：只有价格在区间内的时段计费，出区间自动停止；恰好停在区间边界时仓位
  已是 100% 单边资产、流动性不活跃，同样不计费（计费判断为开区间；
  结果里的 `in_range` 列保持闭区间的旧展示口径）。
  注意 `fee_apr` 应填**本仓位在区间内**的费率——集中流动性会放大单位资金的费率，
  可参考同区间宽度仓位的实际 APR（如 Revert Finance 等工具），不是全池平均 APR。

结果新增列：`fee_income`（当小时）、`total_fee_income`（累积）；
`unhedged_value` / `hedged_value` / `lp_vs_hodl` 均已计入手续费收入。
验证脚本：`python tests/test_fee_income.py`（旧行为回归 + 闭式解 + V3 区间开关）。

## 注意事项
- 请确保数据目录有足够的存储空间
- 建议使用虚拟环境运行项目
- 回测结果仅供参考，不构成投资建议

## 贡献指南
欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 许可证
MIT License 