"""Alpha 因子计算、模型输入列顺序和横截面分位数的滚动聚合。"""

import numpy as np
import pandas as pd

from .Struct import MarketField,MarketQueue
from .WQalpha import FACTORS, FACTOR_LOOKBACKS
from . import OperatorHybrid as op


ALPHA_LIST = [
    'alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006',
    'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013',
    'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha020',
    'alpha021', 'alpha022', 'alpha023', 'alpha025', 'alpha027', 'alpha028',
    'alpha029', 'alpha030', 'alpha033', 'alpha034', 'alpha038', 'alpha040',
    'alpha041', 'alpha042', 'alpha044', 'alpha045', 'alpha046', 'alpha047',
    'alpha049', 'alpha050', 'alpha051', 'alpha054', 'alpha055', 'alpha060',
    'alpha073', 'alpha083', 'alpha101',
]


def factorCal(market : MarketQueue, listed_info, date):
    """将缓存行情复权到计算日，计算 Alpha 因子并对齐当日股票顺序。"""
    history = market.to_frame()
    history['vwap'] = np.nan
    market_tensor = MarketField(history)

    adjustment = op.price_adjustment(market_tensor['close'], market_tensor['prev_close'])
    market_tensor['vwap'][:] = op.safe_divide(
        market_tensor['total_turnover'], market_tensor['volume']
    ) * adjustment
    for field in ('open', 'high', 'low', 'close', 'prev_close'):
        market_tensor[field][:] *= adjustment

    factor_values = np.full((len(market_tensor.assets), len(FACTORS)), np.nan)
    for i, (factor, lookback) in enumerate(zip(FACTORS, FACTOR_LOOKBACKS)):
        if i + 1 != 56 and len(market_tensor.dates) >= lookback:
            factor_values[:, i] = factor(market_tensor)
    factor_values[~(market_tensor['volume'][-1] > 0)] = np.nan

    positions = np.searchsorted(market_tensor.assets, listed_info['order_book_id'].to_numpy())
    factor_names = [f'alpha{i:03d}' for i in range(1, 102)]
    result = pd.DataFrame(factor_values[positions], columns=factor_names)
    result.insert(0, 'order_book_id', listed_info['order_book_id'].to_numpy())
    result.insert(0, 'datetime', date)
    return result


def averRank(factor_summary : MarketField, alpha, WINDOW):
    """每日横截面分位数 -> WINDOW 日等权平均 -> 再排横截面分位数。"""
    rank = pd.DataFrame(factor_summary[alpha][-WINDOW:]).rank(
        axis=1, pct=True
    ).mean(axis=0).rank(pct=True)
    return pd.Series(rank.to_numpy(), index=factor_summary.assets)
