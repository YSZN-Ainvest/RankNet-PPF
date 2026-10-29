"""WorldQuant Alpha101：每个函数接收 MarketField，返回按 mf.assets 排列的当日资产向量。

公式来源 https://arxiv.org/abs/1601.00991 ，Appendix A；非整数窗口向下取整。
调用方保证所需历史长度、字段和数值类型，不在因子内部校验输入。
停牌/缺行用 NaN 占位；时序窗口要求完整，截面排名仅包含有效资产。
returns = close / prev_close - 1；调用方将 OHLC、prev_close、vwap 调整到计算日价格基准。
volume/total_turnover 保留原始交易单位；advN 为 N 日平均成交额（论文定义）。
行业因子暂时直接返回 NaN，待行业数据整合到行情表后再实现。
Alpha056 需要 cap；日行情不含该字段，因子计算时跳过该项。
标准差、协方差 ddof=1；排名并列取平均分位，唯一有效值为 0.5。
"""
import numpy as np
from . import OperatorHybrid as op
FIELD_NAMES = ('open', 'high', 'low', 'close', 'volume', 'total_turnover', 'prev_close')
INDUSTRY_FACTORS = (48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100)


def alpha001(mf):
    """Alpha001：下跌日用收益波动率替代收盘价，取平方峰值位置的截面排序。

    最少 24 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-24:] > 0
    close = np.where(active[-24:], mf['close'][-24:], np.nan)
    prev_close = np.where(active[-24:], mf['prev_close'][-24:], np.nan)
    returns = op.safe_divide(close, prev_close) - 1
    condition1 = op.compare(returns[-5:], 0, np.less)
    stddev1 = op.ts_history(returns, 20, op.ts_stddev, 5)
    selected1 = op.select(condition1, stddev1, close[-5:])
    signed_power1 = op.signed_power(selected1, 2)
    argmax1 = op.ts_argmax(signed_power1, 5)[None, :]
    rank1 = op.rank(argmax1)
    return op.clean((rank1 - 0.5)[0])


def alpha002(mf):
    """Alpha002：成交量对数变化排名与日内收益排名的负相关。

    最少 8 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-8:] > 0
    volume = np.where(active[-8:], mf['volume'][-8:], np.nan)
    log1 = op.log(volume)
    delta1 = op.delta(log1, 2)[-6:]
    rank1 = op.rank(delta1)
    close = np.where(active[-6:], mf['close'][-6:], np.nan)
    open_ = np.where(active[-6:], mf['open'][-6:], np.nan)
    rank2 = op.rank(op.safe_divide(close - open_, open_))
    corr1 = op.ts_corr(rank1, rank2, 6)[None, :]
    return op.clean((-corr1)[0])


def alpha003(mf):
    """Alpha003：开盘价排名与成交量排名的十日负相关。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    open_ = np.where(active[-10:], mf['open'][-10:], np.nan)
    rank1 = op.rank(open_)
    volume = np.where(active[-10:], mf['volume'][-10:], np.nan)
    rank2 = op.rank(volume)
    corr1 = op.ts_corr(rank1, rank2, 10)[None, :]
    return op.clean((-corr1)[0])


def alpha004(mf):
    """Alpha004：最低价截面排名的九日时间序列反向排名。

    最少 9 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-9:] > 0
    low = np.where(active[-9:], mf['low'][-9:], np.nan)
    rank1 = op.rank(low)
    ts_rank1 = op.ts_rank(rank1, 9)[None, :]
    return op.clean((-ts_rank1)[0])


def alpha005(mf):
    """Alpha005：开盘价偏离十日 VWAP 均值与收盘偏离 VWAP 的排名乘积。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    vwap = np.where(active[-10:], mf['vwap'][-10:], np.nan)
    mean1 = op.ts_mean(vwap, 10)[None, :]
    rank1 = op.rank(open_ - mean1)
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    rank2 = op.rank(close - vwap[-1:])
    abs1 = np.abs(rank2)
    return op.clean((rank1 * -abs1)[0])


def alpha006(mf):
    """Alpha006：十日开盘价与成交量的负相关。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    open_ = np.where(active[-10:], mf['open'][-10:], np.nan)
    volume = np.where(active[-10:], mf['volume'][-10:], np.nan)
    corr1 = op.ts_corr(open_, volume, 10)[None, :]
    return op.clean((-corr1)[0])


def alpha007(mf):
    """Alpha007：volume 超过 adv20 时用七日价差反转，否则输出 -1。

    最少 67 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-67:] > 0
    total_turnover = np.where(active[-20:], mf['total_turnover'][-20:], np.nan)
    adv20 = op.ts_mean(total_turnover, 20)[None, :]
    volume = np.where(active[-1:], mf['volume'][-1:], np.nan)
    condition1 = op.compare(adv20, volume, np.less)
    close = np.where(active[-67:], mf['close'][-67:], np.nan)
    delta1 = op.delta(close, 7)[-60:]
    abs1 = np.abs(delta1)
    ts_rank1 = op.ts_rank(abs1, 60)[None, :]
    sign1 = np.sign(delta1[-1:])
    selected1 = op.select(condition1, -ts_rank1 * sign1, -1)
    return op.clean(selected1[0])


def alpha008(mf):
    """Alpha008：五日开盘价和×五日收益和的十日变化反向排名。

    最少 15 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-15:] > 0
    open_ = np.where(active[-15:], mf['open'][-15:], np.nan)
    sum1 = op.ts_history(open_, 5, op.ts_sum, 11)
    close = np.where(active[-15:], mf['close'][-15:], np.nan)
    prev_close = np.where(active[-15:], mf['prev_close'][-15:], np.nan)
    sum2 = op.ts_history(op.safe_divide(close, prev_close) - 1, 5, op.ts_sum, 11)
    delta1 = op.delta(sum1 * sum2, 10)[-1:]
    rank1 = op.rank(delta1)
    return op.clean((-rank1)[0])


def alpha009(mf):
    """Alpha009：连续五日同方向价差沿用方向，否则反转当日价差。

    最少 6 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-6:] > 0
    close = np.where(active[-6:], mf['close'][-6:], np.nan)
    delta1 = op.delta(close, 1)[-5:]
    min1 = op.ts_min(delta1, 5)[None, :]
    condition1 = op.compare(min1, 0, np.greater)
    max1 = op.ts_max(delta1, 5)[None, :]
    condition2 = op.compare(max1, 0, np.less)
    selected1 = op.select(condition2, delta1[-1:], -delta1[-1:])
    selected2 = op.select(condition1, delta1[-1:], selected1)
    return op.clean(selected2[0])


def alpha010(mf):
    """Alpha010：四日连续方向判定后的价差截面排名。

    最少 5 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-5:] > 0
    close = np.where(active[-5:], mf['close'][-5:], np.nan)
    delta1 = op.delta(close, 1)[-4:]
    min1 = op.ts_min(delta1, 4)[None, :]
    condition1 = op.compare(min1, 0, np.greater)
    max1 = op.ts_max(delta1, 4)[None, :]
    condition2 = op.compare(max1, 0, np.less)
    selected1 = op.select(condition2, delta1[-1:], -delta1[-1:])
    selected2 = op.select(condition1, delta1[-1:], selected1)
    rank1 = op.rank(selected2)
    return op.clean(rank1[0])


def alpha011(mf):
    """Alpha011：VWAP/收盘差的三日极值排名与三日量差排名组合。

    最少 4 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-4:] > 0
    vwap = np.where(active[-3:], mf['vwap'][-3:], np.nan)
    close = np.where(active[-3:], mf['close'][-3:], np.nan)
    value1 = vwap - close
    max1 = op.ts_max(value1, 3)[None, :]
    rank1 = op.rank(max1)
    min1 = op.ts_min(value1, 3)[None, :]
    rank2 = op.rank(min1)
    volume = np.where(active[-4:], mf['volume'][-4:], np.nan)
    delta1 = op.delta(volume, 3)[-1:]
    rank3 = op.rank(delta1)
    return op.clean(((rank1 + rank2) * rank3)[0])


def alpha012(mf):
    """Alpha012：成交量变化方向乘以反向收盘价差。

    最少 2 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-2:] > 0
    volume = np.where(active[-2:], mf['volume'][-2:], np.nan)
    delta1 = op.delta(volume, 1)[-1:]
    sign1 = np.sign(delta1)
    close = np.where(active[-2:], mf['close'][-2:], np.nan)
    delta2 = op.delta(close, 1)[-1:]
    return op.clean((sign1 * -delta2)[0])


def alpha013(mf):
    """Alpha013：收盘价/成交量排名的五日协方差反向排名。

    最少 5 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-5:] > 0
    close = np.where(active[-5:], mf['close'][-5:], np.nan)
    rank1 = op.rank(close)
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    rank2 = op.rank(volume)
    cov1 = op.ts_cov(rank1, rank2, 5)[None, :]
    rank3 = op.rank(cov1)
    return op.clean((-rank3)[0])


def alpha014(mf):
    """Alpha014：三日收益变化排名乘以开盘价/成交量相关性。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    close = np.where(active[-4:], mf['close'][-4:], np.nan)
    prev_close = np.where(active[-4:], mf['prev_close'][-4:], np.nan)
    delta1 = op.delta(op.safe_divide(close, prev_close) - 1, 3)[-1:]
    rank1 = op.rank(delta1)
    open_ = np.where(active[-10:], mf['open'][-10:], np.nan)
    volume = np.where(active[-10:], mf['volume'][-10:], np.nan)
    corr1 = op.ts_corr(open_, volume, 10)[None, :]
    return op.clean((-rank1 * corr1)[0])


def alpha015(mf):
    """Alpha015：最高价/成交量排名相关性的排名，作三日负和。

    最少 5 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-5:] > 0
    high = np.where(active[-5:], mf['high'][-5:], np.nan)
    rank1 = op.rank(high)
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    rank2 = op.rank(volume)
    corr1 = op.ts_pair_history(rank1, rank2, 3, op.ts_corr, 3)
    rank3 = op.rank(corr1)
    sum1 = op.ts_sum(rank3, 3)[None, :]
    return op.clean((-sum1)[0])


def alpha016(mf):
    """Alpha016：最高价/成交量排名的五日协方差反向排名。

    最少 5 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-5:] > 0
    high = np.where(active[-5:], mf['high'][-5:], np.nan)
    rank1 = op.rank(high)
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    rank2 = op.rank(volume)
    cov1 = op.ts_cov(rank1, rank2, 5)[None, :]
    rank3 = op.rank(cov1)
    return op.clean((-rank3)[0])


def alpha017(mf):
    """Alpha017：收盘时间排名、价差加速度、相对成交量排名的乘积。

    最少 24 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-24:] > 0
    close = np.where(active[-10:], mf['close'][-10:], np.nan)
    ts_rank1 = op.ts_rank(close, 10)[None, :]
    rank1 = op.rank(ts_rank1)
    delta1 = op.delta(close[-3:], 1)[-2:]
    delta2 = op.delta(delta1, 1)[-1:]
    rank2 = op.rank(delta2)
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    total_turnover = np.where(active[-24:], mf['total_turnover'][-24:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 5)
    ts_rank2 = op.ts_rank(op.safe_divide(volume, adv20), 5)[None, :]
    rank3 = op.rank(ts_rank2)
    return op.clean((-rank1 * rank2 * rank3)[0])


def alpha018(mf):
    """Alpha018：日内价差波动、日内价差及开收相关性的反向排名。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    close = np.where(active[-10:], mf['close'][-10:], np.nan)
    open_ = np.where(active[-10:], mf['open'][-10:], np.nan)
    value1 = close[-5:] - open_[-5:]
    abs1 = np.abs(value1)
    stddev1 = op.ts_stddev(abs1, 5)[None, :]
    corr1 = op.ts_corr(close, open_, 10)[None, :]
    rank1 = op.rank(stddev1 + value1[-1:] + corr1)
    return op.clean((-rank1)[0])


def alpha019(mf):
    """Alpha019：七日价格方向反转，由 250 日收益和排名加权。

    最少 250 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-250:] > 0
    close = np.where(active[-250:], mf['close'][-250:], np.nan)
    delta1 = op.delta(close[-8:], 7)[-1:]
    sign1 = np.sign(delta1 * 2)
    prev_close = np.where(active[-250:], mf['prev_close'][-250:], np.nan)
    sum1 = op.ts_sum(op.safe_divide(close, prev_close) - 1, 250)[None, :]
    rank1 = op.rank(1 + sum1)
    return op.clean((-sign1 * (1 + rank1))[0])


def alpha020(mf):
    """Alpha020：开盘价相对昨日高/收/低价的三个排名负乘积。

    最少 2 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-2:] > 0
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    high = np.where(active[-2:], mf['high'][-2:], np.nan)
    lag1 = high[:-1]
    rank1 = op.rank(open_ - lag1)
    close = np.where(active[-2:], mf['close'][-2:], np.nan)
    lag2 = close[:-1]
    rank2 = op.rank(open_ - lag2)
    low = np.where(active[-2:], mf['low'][-2:], np.nan)
    lag3 = low[:-1]
    rank3 = op.rank(open_ - lag3)
    return op.clean((-rank1 * rank2 * rank3)[0])


def alpha021(mf):
    """Alpha021：均线波动带判定方向，带内由相对成交量判定。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    close = np.where(active[-8:], mf['close'][-8:], np.nan)
    mean1 = op.ts_mean(close, 8)[None, :]
    stddev1 = op.ts_stddev(close, 8)[None, :]
    mean2 = op.ts_mean(close[-2:], 2)[None, :]
    condition1 = op.compare(mean1 + stddev1, mean2, np.less)
    condition2 = op.compare(mean2, mean1 - stddev1, np.less)
    volume = np.where(active[-1:], mf['volume'][-1:], np.nan)
    total_turnover = np.where(active[-20:], mf['total_turnover'][-20:], np.nan)
    adv20 = op.ts_mean(total_turnover, 20)[None, :]
    condition3 = op.compare(op.safe_divide(volume, adv20), 1, np.greater_equal)
    selected1 = op.select(condition3, 1, -1)
    selected2 = op.select(condition2, 1, selected1)
    selected3 = op.select(condition1, -1, selected2)
    return op.clean(selected3[0])


def alpha022(mf):
    """Alpha022：最高价/成交量相关性的五日变化乘收盘波动排名。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    high = np.where(active[-10:], mf['high'][-10:], np.nan)
    volume = np.where(active[-10:], mf['volume'][-10:], np.nan)
    corr1 = op.ts_pair_history(high, volume, 5, op.ts_corr, 6)
    delta1 = op.delta(corr1, 5)[-1:]
    close = np.where(active[-20:], mf['close'][-20:], np.nan)
    stddev1 = op.ts_stddev(close, 20)[None, :]
    rank1 = op.rank(stddev1)
    return op.clean((-delta1 * rank1)[0])


def alpha023(mf):
    """Alpha023：最高价高于二十日均值时反转两日高价变化，否则为零。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    high = np.where(active[-20:], mf['high'][-20:], np.nan)
    mean1 = op.ts_mean(high, 20)[None, :]
    condition1 = op.compare(mean1, high[-1:], np.less)
    delta1 = op.delta(high[-3:], 2)[-1:]
    selected1 = op.select(condition1, -delta1, 0)
    return op.clean(selected1[0])


def alpha024(mf):
    """Alpha024：百日均线增速较低时使用低点偏离，否则使用三日反转。

    最少 200 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-200:] > 0
    close = np.where(active[-200:], mf['close'][-200:], np.nan)
    mean1 = op.ts_history(close, 100, op.ts_mean, 101)
    delta1 = op.delta(mean1, 100)[-1:]
    lag1 = close[-101:][:-100]
    condition1 = op.compare(op.safe_divide(delta1, lag1), 0.05, np.less_equal)
    min1 = op.ts_min(close[-100:], 100)[None, :]
    delta2 = op.delta(close[-4:], 3)[-1:]
    selected1 = op.select(condition1, -(close[-1:] - min1), -delta2)
    return op.clean(selected1[0])


def alpha025(mf):
    """Alpha025：反向收益、adv20、VWAP 和收盘距最高价的乘积排名。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    prev_close = np.where(active[-1:], mf['prev_close'][-1:], np.nan)
    total_turnover = np.where(active[-20:], mf['total_turnover'][-20:], np.nan)
    adv20 = op.ts_mean(total_turnover, 20)[None, :]
    vwap = np.where(active[-1:], mf['vwap'][-1:], np.nan)
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    rank1 = op.rank(-(op.safe_divide(close, prev_close) - 1) * adv20 * vwap * (high - close))
    return op.clean(rank1[0])


def alpha026(mf):
    """Alpha026：成交量/最高价时间排名相关性的三日最大值取负。

    最少 11 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-11:] > 0
    volume = np.where(active[-11:], mf['volume'][-11:], np.nan)
    ts_rank1 = op.ts_history(volume, 5, op.ts_rank, 7)
    high = np.where(active[-11:], mf['high'][-11:], np.nan)
    ts_rank2 = op.ts_history(high, 5, op.ts_rank, 7)
    corr1 = op.ts_pair_history(ts_rank1, ts_rank2, 5, op.ts_corr, 3)
    max1 = op.ts_max(corr1, 3)[None, :]
    return op.clean((-max1)[0])


def alpha027(mf):
    """Alpha027：量价排名相关性的两日均值排名，按 0.5 切分方向。

    最少 7 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-7:] > 0
    volume = np.where(active[-7:], mf['volume'][-7:], np.nan)
    rank1 = op.rank(volume)
    vwap = np.where(active[-7:], mf['vwap'][-7:], np.nan)
    rank2 = op.rank(vwap)
    corr1 = op.ts_pair_history(rank1, rank2, 6, op.ts_corr, 2)
    mean1 = op.ts_mean(corr1, 2)[None, :]
    rank3 = op.rank(mean1)
    condition1 = op.compare(rank3, 0.5, np.greater)
    selected1 = op.select(condition1, -1, 1)
    return op.clean(selected1[0])


def alpha028(mf):
    """Alpha028：adv20/低价相关性加中间价与收盘价差，截面缩放。

    最少 24 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-24:] > 0
    total_turnover = np.where(active[-24:], mf['total_turnover'][-24:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 5)
    low = np.where(active[-5:], mf['low'][-5:], np.nan)
    corr1 = op.ts_corr(adv20, low, 5)[None, :]
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    scale1 = op.scale(corr1 + op.safe_divide(high + low[-1:], 2) - close)
    return op.clean(scale1[0])


def alpha029(mf):
    """Alpha029：嵌套价差排名与延迟反向收益排名；外层 min 是五日时序最小值。

    最少 11 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-11:] > 0
    close = np.where(active[-11:], mf['close'][-11:], np.nan)
    delta1 = op.delta(close - 1, 5)[-6:]
    rank1 = op.rank(delta1)
    rank2 = op.rank(-rank1)
    rank3 = op.rank(rank2)
    min1 = op.ts_history(rank3, 2, op.ts_min, 5)
    sum1 = op.ts_history(min1, 1, op.ts_sum, 5)
    log1 = op.log(sum1)
    scale1 = op.scale(log1)
    rank4 = op.rank(scale1)
    rank5 = op.rank(rank4)
    product1 = op.ts_history(rank5, 1, op.ts_product, 5)
    min2 = op.ts_min(product1, 5)[None, :]
    prev_close = np.where(active[-11:], mf['prev_close'][-11:], np.nan)
    lag1 = (-(op.safe_divide(close, prev_close) - 1))[:-6]
    ts_rank1 = op.ts_rank(lag1, 5)[None, :]
    return op.clean((min2 + ts_rank1)[0])


def alpha030(mf):
    """Alpha030：三日涨跌方向和的反向排名，用五/二十日成交量和比加权。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    close = np.where(active[-4:], mf['close'][-4:], np.nan)
    delta1 = op.delta(close, 1)[-3:]
    sign1 = np.sign(delta1[-1:])
    lag1 = delta1[-2:][:-1]
    sign2 = np.sign(lag1)
    lag2 = delta1[:-2]
    sign3 = np.sign(lag2)
    rank1 = op.rank(sign1 + sign2 + sign3)
    volume = np.where(active[-20:], mf['volume'][-20:], np.nan)
    sum1 = op.ts_sum(volume[-5:], 5)[None, :]
    sum2 = op.ts_sum(volume, 20)[None, :]
    return op.clean(op.safe_divide((1 - rank1) * sum1, sum2)[0])


def alpha031(mf):
    """Alpha031：衰减价差排名、短期反转和 adv20/低价相关方向的叠加。

    最少 31 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-31:] > 0
    close = np.where(active[-20:], mf['close'][-20:], np.nan)
    delta1 = op.delta(close, 10)[-10:]
    rank1 = op.rank(delta1)
    rank2 = op.rank(rank1)
    decay1 = op.ts_decay_linear(-rank2, 10)[None, :]
    rank3 = op.rank(decay1)
    rank4 = op.rank(rank3)
    rank5 = op.rank(rank4)
    delta2 = op.delta(close[-4:], 3)[-1:]
    rank6 = op.rank(-delta2)
    total_turnover = np.where(active[-31:], mf['total_turnover'][-31:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 12)
    low = np.where(active[-12:], mf['low'][-12:], np.nan)
    corr1 = op.ts_corr(adv20, low, 12)[None, :]
    scale1 = op.scale(corr1)
    sign1 = np.sign(scale1)
    return op.clean((rank5 + rank6 + sign1)[0])


def alpha032(mf):
    """Alpha032：七日均线偏离与 VWAP/延迟收盘的长窗口相关性。

    最少 235 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-235:] > 0
    close = np.where(active[-235:], mf['close'][-235:], np.nan)
    mean1 = op.ts_mean(close[-7:], 7)[None, :]
    scale1 = op.scale(mean1 - close[-1:])
    vwap = np.where(active[-230:], mf['vwap'][-230:], np.nan)
    lag1 = close[:-5]
    corr1 = op.ts_corr(vwap, lag1, 230)[None, :]
    scale2 = op.scale(corr1)
    return op.clean((scale1 + 20 * scale2)[0])


def alpha033(mf):
    """Alpha033：开盘/收盘比减一的截面排名。

    最少 1 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-1:] > 0
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    rank1 = op.rank(op.safe_divide(open_, close) - 1)
    return op.clean(rank1[0])


def alpha034(mf):
    """Alpha034：短长收益波动比和单日价差的反向排名组合。

    最少 5 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-5:] > 0
    close = np.where(active[-5:], mf['close'][-5:], np.nan)
    prev_close = np.where(active[-5:], mf['prev_close'][-5:], np.nan)
    returns = op.safe_divide(close, prev_close) - 1
    stddev1 = op.ts_stddev(returns[-2:], 2)[None, :]
    stddev2 = op.ts_stddev(returns, 5)[None, :]
    rank1 = op.rank(op.safe_divide(stddev1, stddev2))
    delta1 = op.delta(close[-2:], 1)[-1:]
    rank2 = op.rank(delta1)
    rank3 = op.rank(1 - rank1 + 1 - rank2)
    return op.clean(rank3[0])


def alpha035(mf):
    """Alpha035：成交量、收盘加振幅、收益率三个时间排名的乘积。

    最少 32 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-32:] > 0
    volume = np.where(active[-32:], mf['volume'][-32:], np.nan)
    ts_rank1 = op.ts_rank(volume, 32)[None, :]
    close = np.where(active[-32:], mf['close'][-32:], np.nan)
    high = np.where(active[-16:], mf['high'][-16:], np.nan)
    low = np.where(active[-16:], mf['low'][-16:], np.nan)
    ts_rank2 = op.ts_rank(close[-16:] + high - low, 16)[None, :]
    prev_close = np.where(active[-32:], mf['prev_close'][-32:], np.nan)
    ts_rank3 = op.ts_rank(op.safe_divide(close, prev_close) - 1, 32)[None, :]
    return op.clean((ts_rank1 * (1 - ts_rank2) * (1 - ts_rank3))[0])


def alpha036(mf):
    """Alpha036：五项价量、反转和长期均线偏离信号的原式加权组合。

    最少 200 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-200:] > 0
    close = np.where(active[-200:], mf['close'][-200:], np.nan)
    open_ = np.where(active[-15:], mf['open'][-15:], np.nan)
    value1 = close[-15:] - open_
    volume = np.where(active[-16:], mf['volume'][-16:], np.nan)
    lag1 = volume[:-1]
    corr1 = op.ts_corr(value1, lag1, 15)[None, :]
    rank1 = op.rank(corr1)
    rank2 = op.rank(open_[-1:] - close[-1:])
    prev_close = np.where(active[-11:], mf['prev_close'][-11:], np.nan)
    lag2 = (-(op.safe_divide(close[-11:], prev_close) - 1))[:-6]
    ts_rank1 = op.ts_rank(lag2, 5)[None, :]
    rank3 = op.rank(ts_rank1)
    vwap = np.where(active[-6:], mf['vwap'][-6:], np.nan)
    total_turnover = np.where(active[-25:], mf['total_turnover'][-25:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 6)
    corr2 = op.ts_corr(vwap, adv20, 6)[None, :]
    abs1 = np.abs(corr2)
    rank4 = op.rank(abs1)
    mean2 = op.ts_mean(close, 200)[None, :]
    rank5 = op.rank((mean2 - open_[-1:]) * value1[-1:])
    return op.clean((2.21 * rank1 + 0.7 * rank2 + 0.73 * rank3 + rank4 + 0.6 * rank5)[0])


def alpha037(mf):
    """Alpha037：昨日开收价差与收盘价的长期相关排名，加当日开收价差排名。

    最少 201 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-201:] > 0
    open_ = np.where(active[-201:], mf['open'][-201:], np.nan)
    close = np.where(active[-201:], mf['close'][-201:], np.nan)
    value1 = open_ - close
    lag1 = value1[:-1]
    corr1 = op.ts_corr(lag1, close[-200:], 200)[None, :]
    rank1 = op.rank(corr1)
    rank2 = op.rank(value1[-1:])
    return op.clean((rank1 + rank2)[0])


def alpha038(mf):
    """Alpha038：收盘时间排名的负截面排名，乘收开比排名。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    close = np.where(active[-10:], mf['close'][-10:], np.nan)
    ts_rank1 = op.ts_rank(close, 10)[None, :]
    rank1 = op.rank(ts_rank1)
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    rank2 = op.rank(op.safe_divide(close[-1:], open_))
    return op.clean((-rank1 * rank2)[0])


def alpha039(mf):
    """Alpha039：量比衰减加权的七日反转，再由长期收益排名加权。

    最少 250 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-250:] > 0
    close = np.where(active[-250:], mf['close'][-250:], np.nan)
    delta1 = op.delta(close[-8:], 7)[-1:]
    volume = np.where(active[-9:], mf['volume'][-9:], np.nan)
    total_turnover = np.where(active[-28:], mf['total_turnover'][-28:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 9)
    decay1 = op.ts_decay_linear(op.safe_divide(volume, adv20), 9)[None, :]
    rank1 = op.rank(decay1)
    rank2 = op.rank(delta1 * (1 - rank1))
    prev_close = np.where(active[-250:], mf['prev_close'][-250:], np.nan)
    sum1 = op.ts_sum(op.safe_divide(close, prev_close) - 1, 250)[None, :]
    rank3 = op.rank(sum1)
    return op.clean((-rank2 * (1 + rank3))[0])


def alpha040(mf):
    """Alpha040：最高价波动排名乘最高价/成交量相关性，取负。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    high = np.where(active[-10:], mf['high'][-10:], np.nan)
    stddev1 = op.ts_stddev(high, 10)[None, :]
    rank1 = op.rank(stddev1)
    volume = np.where(active[-10:], mf['volume'][-10:], np.nan)
    corr1 = op.ts_corr(high, volume, 10)[None, :]
    return op.clean((-rank1 * corr1)[0])


def alpha041(mf):
    """Alpha041：高低价几何均值减 VWAP。

    最少 1 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-1:] > 0
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    low = np.where(active[-1:], mf['low'][-1:], np.nan)
    vwap = np.where(active[-1:], mf['vwap'][-1:], np.nan)
    return op.clean((op.power(high * low, 0.5) - vwap)[0])


def alpha042(mf):
    """Alpha042：VWAP/收盘价差排名除以两者和的排名。

    最少 1 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-1:] > 0
    vwap = np.where(active[-1:], mf['vwap'][-1:], np.nan)
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    rank1 = op.rank(vwap - close)
    rank2 = op.rank(vwap + close)
    return op.clean(op.safe_divide(rank1, rank2)[0])


def alpha043(mf):
    """Alpha043：相对成交量排名与七日反向价差排名的乘积。

    最少 39 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-39:] > 0
    volume = np.where(active[-20:], mf['volume'][-20:], np.nan)
    total_turnover = np.where(active[-39:], mf['total_turnover'][-39:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 20)
    ts_rank1 = op.ts_rank(op.safe_divide(volume, adv20), 20)[None, :]
    close = np.where(active[-15:], mf['close'][-15:], np.nan)
    delta1 = op.delta(close, 7)[-8:]
    ts_rank2 = op.ts_rank(-delta1, 8)[None, :]
    return op.clean((ts_rank1 * ts_rank2)[0])


def alpha044(mf):
    """Alpha044：最高价与成交量截面排名的五日负相关。

    最少 5 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-5:] > 0
    high = np.where(active[-5:], mf['high'][-5:], np.nan)
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    rank1 = op.rank(volume)
    corr1 = op.ts_corr(high, rank1, 5)[None, :]
    return op.clean((-corr1)[0])


def alpha045(mf):
    """Alpha045：延迟均价排名、价量相关与短长价格和相关排名的负乘积。

    最少 25 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-25:] > 0
    close = np.where(active[-25:], mf['close'][-25:], np.nan)
    lag1 = close[:-5]
    mean1 = op.ts_mean(lag1, 20)[None, :]
    rank1 = op.rank(mean1)
    volume = np.where(active[-2:], mf['volume'][-2:], np.nan)
    corr1 = op.ts_corr(close[-2:], volume, 2)[None, :]
    sum1 = op.ts_history(close[-6:], 5, op.ts_sum, 2)
    sum2 = op.ts_history(close[-21:], 20, op.ts_sum, 2)
    corr2 = op.ts_corr(sum1, sum2, 2)[None, :]
    rank2 = op.rank(corr2)
    return op.clean((-rank1 * corr1 * rank2)[0])


def alpha046(mf):
    """Alpha046：十日斜率差超过阈值时输出方向，否则使用单日反转。

    最少 21 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-21:] > 0
    close = np.where(active[-21:], mf['close'][-21:], np.nan)
    lag1 = close[:-20]
    lag2 = close[-11:][:-10]
    value5 = op.safe_divide(lag1 - lag2, 10) - op.safe_divide(lag2 - close[-1:], 10)
    condition1 = op.compare(value5, 0.25, np.greater)
    condition2 = op.compare(value5, 0, np.less)
    delta1 = op.delta(close[-2:], 1)[-1:]
    selected1 = op.select(condition2, 1, -delta1)
    selected2 = op.select(condition1, -1, selected1)
    return op.clean(selected2[0])


def alpha047(mf):
    """Alpha047：倒数价格排名、相对成交量及高价偏离组合，减 VWAP 动量排名。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    rank1 = op.rank(op.safe_divide(1, close))
    volume = np.where(active[-1:], mf['volume'][-1:], np.nan)
    total_turnover = np.where(active[-20:], mf['total_turnover'][-20:], np.nan)
    adv20 = op.ts_mean(total_turnover, 20)[None, :]
    high = np.where(active[-5:], mf['high'][-5:], np.nan)
    rank2 = op.rank(high[-1:] - close)
    mean2 = op.ts_mean(high, 5)[None, :]
    vwap = np.where(active[-6:], mf['vwap'][-6:], np.nan)
    delta1 = op.delta(vwap, 5)[-1:]
    rank3 = op.rank(delta1)
    value8 = op.safe_divide(rank1 * op.safe_divide(volume, adv20) * high[-1:] * rank2, mean2) - rank3
    return op.clean(value8[0])


def alpha048(mf):
    """Alpha048：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha049(mf):
    """Alpha049：价格斜率差，低于 -0.1 时为1，否则单日反转。

    最少 21 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-21:] > 0
    close = np.where(active[-21:], mf['close'][-21:], np.nan)
    lag1 = close[:-20]
    lag2 = close[-11:][:-10]
    condition1 = op.compare(op.safe_divide(lag1 - lag2, 10) - op.safe_divide(lag2 - close[-1:], 10), -0.1, np.less)
    delta1 = op.delta(close[-2:], 1)[-1:]
    selected1 = op.select(condition1, 1, -delta1)
    return op.clean(selected1[0])


def alpha050(mf):
    """Alpha050：成交量/VWAP 排名相关性排名的五日最大值取负。

    最少 9 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-9:] > 0
    volume = np.where(active[-9:], mf['volume'][-9:], np.nan)
    rank1 = op.rank(volume)
    vwap = np.where(active[-9:], mf['vwap'][-9:], np.nan)
    rank2 = op.rank(vwap)
    corr1 = op.ts_pair_history(rank1, rank2, 5, op.ts_corr, 5)
    rank3 = op.rank(corr1)
    max1 = op.ts_max(rank3, 5)[None, :]
    return op.clean((-max1)[0])


def alpha051(mf):
    """Alpha051：价格斜率差，低于 -0.05 时为1，否则单日反转。

    最少 21 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-21:] > 0
    close = np.where(active[-21:], mf['close'][-21:], np.nan)
    lag1 = close[:-20]
    lag2 = close[-11:][:-10]
    condition1 = op.compare(op.safe_divide(lag1 - lag2, 10) - op.safe_divide(lag2 - close[-1:], 10), -0.05, np.less)
    delta1 = op.delta(close[-2:], 1)[-1:]
    selected1 = op.select(condition1, 1, -delta1)
    return op.clean(selected1[0])


def alpha052(mf):
    """Alpha052：低点变化、排除最近二十日的长期收益排名和成交量排名组合。

    最少 240 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-240:] > 0
    low = np.where(active[-10:], mf['low'][-10:], np.nan)
    min1 = op.ts_history(low, 5, op.ts_min, 6)
    lag1 = min1[:-5]
    close = np.where(active[-240:], mf['close'][-240:], np.nan)
    prev_close = np.where(active[-240:], mf['prev_close'][-240:], np.nan)
    returns = op.safe_divide(close, prev_close) - 1
    sum1 = op.ts_sum(returns, 240)[None, :]
    sum2 = op.ts_sum(returns[-20:], 20)[None, :]
    rank1 = op.rank(op.safe_divide(sum1 - sum2, 220))
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    ts_rank1 = op.ts_rank(volume, 5)[None, :]
    return op.clean(((lag1 - min1[-1:]) * rank1 * ts_rank1)[0])


def alpha053(mf):
    """Alpha053：收盘在高低区间位置比的九日变化取负。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    close = np.where(active[-10:], mf['close'][-10:], np.nan)
    low = np.where(active[-10:], mf['low'][-10:], np.nan)
    value1 = close - low
    high = np.where(active[-10:], mf['high'][-10:], np.nan)
    delta1 = op.delta(op.safe_divide(value1 - (high - close), value1), 9)[-1:]
    return op.clean((-delta1)[0])


def alpha054(mf):
    """Alpha054：低收价差与开盘五次幂之积，除以低高差与收盘五次幂。

    最少 1 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-1:] > 0
    low = np.where(active[-1:], mf['low'][-1:], np.nan)
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    return op.clean(op.safe_divide(-(low - close) * op.power(open_, 5), (low - high) * op.power(close, 5))[0])


def alpha055(mf):
    """Alpha055：十二日价格区间位置排名与成交量排名的负相关。

    最少 17 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-17:] > 0
    close = np.where(active[-6:], mf['close'][-6:], np.nan)
    low = np.where(active[-17:], mf['low'][-17:], np.nan)
    min1 = op.ts_history(low, 12, op.ts_min, 6)
    high = np.where(active[-17:], mf['high'][-17:], np.nan)
    max1 = op.ts_history(high, 12, op.ts_max, 6)
    rank1 = op.rank(op.safe_divide(close - min1, max1 - min1))
    volume = np.where(active[-6:], mf['volume'][-6:], np.nan)
    rank2 = op.rank(volume)
    corr1 = op.ts_corr(rank1, rank2, 6)[None, :]
    return op.clean((-corr1)[0])


def alpha056(mf):
    """Alpha056：收益和比排名乘以收益×市值排名，取负；需要 cap。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    close = np.where(active[-10:], mf['close'][-10:], np.nan)
    prev_close = np.where(active[-10:], mf['prev_close'][-10:], np.nan)
    returns = op.safe_divide(close, prev_close) - 1
    sum1 = op.ts_sum(returns, 10)[None, :]
    sum2 = op.ts_history(returns[-4:], 2, op.ts_sum, 3)
    sum3 = op.ts_sum(sum2, 3)[None, :]
    rank1 = op.rank(op.safe_divide(sum1, sum3))
    cap = np.where(active[-1:], mf['cap'][-1:], np.nan)
    rank2 = op.rank(returns[-1:] * cap)
    return op.clean((-rank1 * rank2)[0])


def alpha057(mf):
    """Alpha057：收盘/VWAP 价差除以衰减后的三十日高点位置排名，取负。

    最少 31 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-31:] > 0
    close = np.where(active[-31:], mf['close'][-31:], np.nan)
    vwap = np.where(active[-1:], mf['vwap'][-1:], np.nan)
    argmax1 = op.ts_history(close, 30, op.ts_argmax, 2)
    rank1 = op.rank(argmax1)
    decay1 = op.ts_decay_linear(rank1, 2)[None, :]
    return op.clean(op.safe_divide(-(close[-1:] - vwap), decay1)[0])


def alpha058(mf):
    """Alpha058：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha059(mf):
    """Alpha059：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha060(mf):
    """Alpha060：成交量加权区间位置排名与十日高点位置排名的缩放差。

    最少 10 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-10:] > 0
    close = np.where(active[-10:], mf['close'][-10:], np.nan)
    low = np.where(active[-1:], mf['low'][-1:], np.nan)
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    volume = np.where(active[-1:], mf['volume'][-1:], np.nan)
    rank1 = op.rank(op.safe_divide(close[-1:] - low - (high - close[-1:]), high - low) * volume)
    scale1 = op.scale(rank1)
    argmax1 = op.ts_argmax(close, 10)[None, :]
    rank2 = op.rank(argmax1)
    scale2 = op.scale(rank2)
    return op.clean((-(2 * scale1 - scale2))[0])


def alpha061(mf):
    """Alpha061：比较 VWAP 距低点排名和 VWAP/adv180 相关性排名，输出0/1。

    最少 196 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-196:] > 0
    vwap = np.where(active[-17:], mf['vwap'][-17:], np.nan)
    min1 = op.ts_min(vwap[-16:], 16)[None, :]
    rank1 = op.rank(vwap[-1:] - min1)
    total_turnover = np.where(active[-196:], mf['total_turnover'][-196:], np.nan)
    adv180 = op.ts_history(total_turnover, 180, op.ts_mean, 17)
    corr1 = op.ts_corr(vwap, adv180, 17)[None, :]
    rank2 = op.rank(corr1)
    condition1 = op.compare(rank1, rank2, np.less)
    return op.clean(condition1[0])


def alpha062(mf):
    """Alpha062：VWAP/累计流动性相关排名与开盘相对高低价排名条件比较。

    最少 49 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-49:] > 0
    vwap = np.where(active[-9:], mf['vwap'][-9:], np.nan)
    total_turnover = np.where(active[-49:], mf['total_turnover'][-49:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 30)
    sum1 = op.ts_history(adv20, 22, op.ts_sum, 9)
    corr1 = op.ts_corr(vwap, sum1, 9)[None, :]
    rank1 = op.rank(corr1)
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    rank2 = op.rank(open_)
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    low = np.where(active[-1:], mf['low'][-1:], np.nan)
    rank3 = op.rank(op.safe_divide(high + low, 2))
    rank4 = op.rank(high)
    condition1 = op.compare(rank2 * 2, rank3 + rank4, np.less)
    rank5 = op.rank(condition1)
    condition2 = op.compare(rank1, rank5, np.less)
    return op.clean((-condition2)[0])


def alpha063(mf):
    """Alpha063：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha064(mf):
    """Alpha064：混合价格和/累计 adv120 相关性与混合价格变化的排名比较。

    最少 146 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-146:] > 0
    open_ = np.where(active[-27:], mf['open'][-27:], np.nan)
    low = np.where(active[-27:], mf['low'][-27:], np.nan)
    sum1 = op.ts_history(open_ * 0.178404 + low * 0.821596, 12, op.ts_sum, 16)
    total_turnover = np.where(active[-146:], mf['total_turnover'][-146:], np.nan)
    adv120 = op.ts_history(total_turnover, 120, op.ts_mean, 27)
    sum2 = op.ts_history(adv120, 12, op.ts_sum, 16)
    corr1 = op.ts_corr(sum1, sum2, 16)[None, :]
    rank1 = op.rank(corr1)
    high = np.where(active[-4:], mf['high'][-4:], np.nan)
    vwap = np.where(active[-4:], mf['vwap'][-4:], np.nan)
    delta1 = op.delta(op.safe_divide(high + low[-4:], 2) * 0.178404 + vwap * 0.821596, 3)[-1:]
    rank2 = op.rank(delta1)
    condition1 = op.compare(rank1, rank2, np.less)
    return op.clean((-condition1)[0])


def alpha065(mf):
    """Alpha065：开盘/VWAP 混合价与累计 adv60 的相关排名，比较开盘距低点排名。

    最少 72 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-72:] > 0
    open_ = np.where(active[-13:], mf['open'][-13:], np.nan)
    vwap = np.where(active[-6:], mf['vwap'][-6:], np.nan)
    total_turnover = np.where(active[-72:], mf['total_turnover'][-72:], np.nan)
    adv60 = op.ts_history(total_turnover, 60, op.ts_mean, 13)
    sum1 = op.ts_history(adv60, 8, op.ts_sum, 6)
    corr1 = op.ts_corr(open_[-6:] * 0.00817205 + vwap * 0.99182795, sum1, 6)[None, :]
    rank1 = op.rank(corr1)
    min1 = op.ts_min(open_, 13)[None, :]
    rank2 = op.rank(open_[-1:] - min1)
    condition1 = op.compare(rank1, rank2, np.less)
    return op.clean((-condition1)[0])


def alpha066(mf):
    """Alpha066：VWAP 变化与低价/VWAP 偏离比的衰减排名之和取负。

    最少 16 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-16:] > 0
    vwap = np.where(active[-16:], mf['vwap'][-16:], np.nan)
    delta1 = op.delta(vwap[-10:], 3)[-7:]
    decay1 = op.ts_decay_linear(delta1, 7)[None, :]
    rank1 = op.rank(decay1)
    low = np.where(active[-16:], mf['low'][-16:], np.nan)
    open_ = np.where(active[-16:], mf['open'][-16:], np.nan)
    high = np.where(active[-16:], mf['high'][-16:], np.nan)
    decay2 = op.ts_history(op.safe_divide(low - vwap, open_ - op.safe_divide(high + low, 2)), 11, op.ts_decay_linear, 6)
    ts_rank1 = op.ts_rank(decay2, 6)[None, :]
    return op.clean((-(rank1 + ts_rank1))[0])


def alpha067(mf):
    """Alpha067：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha068(mf):
    """Alpha068：高价/adv15 排名相关的时间排名，与混合收低价变化排名比较。

    最少 34 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-34:] > 0
    high = np.where(active[-20:], mf['high'][-20:], np.nan)
    rank1 = op.rank(high)
    total_turnover = np.where(active[-34:], mf['total_turnover'][-34:], np.nan)
    adv15 = op.ts_history(total_turnover, 15, op.ts_mean, 20)
    rank2 = op.rank(adv15)
    corr1 = op.ts_pair_history(rank1, rank2, 8, op.ts_corr, 13)
    ts_rank1 = op.ts_rank(corr1, 13)[None, :]
    close = np.where(active[-2:], mf['close'][-2:], np.nan)
    low = np.where(active[-2:], mf['low'][-2:], np.nan)
    delta1 = op.delta(close * 0.518371 + low * 0.481629, 1)[-1:]
    rank3 = op.rank(delta1)
    condition1 = op.compare(ts_rank1, rank3, np.less)
    return op.clean((-condition1)[0])


def alpha069(mf):
    """Alpha069：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha070(mf):
    """Alpha070：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha071(mf):
    """Alpha071：量价时间排名相关信号与低开价偏离 VWAP 信号，逐元素取大。

    最少 225 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-225:] > 0
    close = np.where(active[-37:], mf['close'][-37:], np.nan)
    ts_rank1 = op.ts_history(close, 3, op.ts_rank, 35)
    total_turnover = np.where(active[-225:], mf['total_turnover'][-225:], np.nan)
    adv180 = op.ts_history(total_turnover, 180, op.ts_mean, 46)
    ts_rank2 = op.ts_history(adv180, 12, op.ts_rank, 35)
    corr1 = op.ts_pair_history(ts_rank1, ts_rank2, 18, op.ts_corr, 18)
    decay1 = op.ts_history(corr1, 4, op.ts_decay_linear, 15)
    ts_rank3 = op.ts_rank(decay1, 15)[None, :]
    low = np.where(active[-19:], mf['low'][-19:], np.nan)
    open_ = np.where(active[-19:], mf['open'][-19:], np.nan)
    vwap = np.where(active[-19:], mf['vwap'][-19:], np.nan)
    rank1 = op.rank(low + open_ - 2 * vwap)
    decay2 = op.ts_history(op.power(rank1, 2), 16, op.ts_decay_linear, 4)
    ts_rank4 = op.ts_rank(decay2, 4)[None, :]
    maximum1 = np.maximum(ts_rank3, ts_rank4)
    return op.clean(maximum1[0])


def alpha072(mf):
    """Alpha072：中间价/adv40 衰减相关排名，除以 VWAP/成交量时间排名相关信号。

    最少 56 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-56:] > 0
    high = np.where(active[-17:], mf['high'][-17:], np.nan)
    low = np.where(active[-17:], mf['low'][-17:], np.nan)
    total_turnover = np.where(active[-56:], mf['total_turnover'][-56:], np.nan)
    adv40 = op.ts_history(total_turnover, 40, op.ts_mean, 17)
    corr1 = op.ts_pair_history(op.safe_divide(high + low, 2), adv40, 8, op.ts_corr, 10)
    decay1 = op.ts_decay_linear(corr1, 10)[None, :]
    rank1 = op.rank(decay1)
    vwap = np.where(active[-9:], mf['vwap'][-9:], np.nan)
    ts_rank1 = op.ts_history(vwap, 3, op.ts_rank, 7)
    volume = np.where(active[-24:], mf['volume'][-24:], np.nan)
    ts_rank2 = op.ts_history(volume, 18, op.ts_rank, 7)
    corr2 = op.ts_pair_history(ts_rank1, ts_rank2, 6, op.ts_corr, 2)
    decay2 = op.ts_decay_linear(corr2, 2)[None, :]
    rank2 = op.rank(decay2)
    return op.clean(op.safe_divide(rank1, rank2)[0])


def alpha073(mf):
    """Alpha073：VWAP 衰减变化和混合价格相对变化两个排名，逐元素取大后取负。

    最少 20 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-20:] > 0
    vwap = np.where(active[-6:], mf['vwap'][-6:], np.nan)
    delta1 = op.delta(vwap, 4)[-2:]
    decay1 = op.ts_decay_linear(delta1, 2)[None, :]
    rank1 = op.rank(decay1)
    open_ = np.where(active[-20:], mf['open'][-20:], np.nan)
    low = np.where(active[-20:], mf['low'][-20:], np.nan)
    value3 = open_ * 0.147155 + low * 0.852845
    delta2 = op.delta(value3, 2)[-18:]
    decay2 = op.ts_history(op.safe_divide(-delta2, value3[-18:]), 3, op.ts_decay_linear, 16)
    ts_rank1 = op.ts_rank(decay2, 16)[None, :]
    maximum1 = np.maximum(rank1, ts_rank1)
    return op.clean((-maximum1)[0])


def alpha074(mf):
    """Alpha074：收盘/累计 adv30 相关排名与混合高价/VWAP 量价排名相关比较。

    最少 80 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-80:] > 0
    close = np.where(active[-15:], mf['close'][-15:], np.nan)
    total_turnover = np.where(active[-80:], mf['total_turnover'][-80:], np.nan)
    adv30 = op.ts_history(total_turnover, 30, op.ts_mean, 51)
    sum1 = op.ts_history(adv30, 37, op.ts_sum, 15)
    corr1 = op.ts_corr(close, sum1, 15)[None, :]
    rank1 = op.rank(corr1)
    high = np.where(active[-11:], mf['high'][-11:], np.nan)
    vwap = np.where(active[-11:], mf['vwap'][-11:], np.nan)
    rank2 = op.rank(high * 0.0261661 + vwap * 0.9738339)
    volume = np.where(active[-11:], mf['volume'][-11:], np.nan)
    rank3 = op.rank(volume)
    corr2 = op.ts_corr(rank2, rank3, 11)[None, :]
    rank4 = op.rank(corr2)
    condition1 = op.compare(rank1, rank4, np.less)
    return op.clean((-condition1)[0])


def alpha075(mf):
    """Alpha075：VWAP/成交量相关排名与低价/adv50 排名相关排名比较。

    最少 61 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-61:] > 0
    vwap = np.where(active[-4:], mf['vwap'][-4:], np.nan)
    volume = np.where(active[-4:], mf['volume'][-4:], np.nan)
    corr1 = op.ts_corr(vwap, volume, 4)[None, :]
    rank1 = op.rank(corr1)
    low = np.where(active[-12:], mf['low'][-12:], np.nan)
    rank2 = op.rank(low)
    total_turnover = np.where(active[-61:], mf['total_turnover'][-61:], np.nan)
    adv50 = op.ts_history(total_turnover, 50, op.ts_mean, 12)
    rank3 = op.rank(adv50)
    corr2 = op.ts_corr(rank2, rank3, 12)[None, :]
    rank4 = op.rank(corr2)
    condition1 = op.compare(rank1, rank4, np.less)
    return op.clean(condition1[0])


def alpha076(mf):
    """Alpha076：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha077(mf):
    """Alpha077：中间价/VWAP 偏离与中间价/adv40 相关的衰减排名，取小。

    最少 46 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-46:] > 0
    high = np.where(active[-20:], mf['high'][-20:], np.nan)
    low = np.where(active[-20:], mf['low'][-20:], np.nan)
    value2 = op.safe_divide(high + low, 2)
    vwap = np.where(active[-20:], mf['vwap'][-20:], np.nan)
    decay1 = op.ts_decay_linear(value2 - vwap, 20)[None, :]
    rank1 = op.rank(decay1)
    total_turnover = np.where(active[-46:], mf['total_turnover'][-46:], np.nan)
    adv40 = op.ts_history(total_turnover, 40, op.ts_mean, 7)
    corr1 = op.ts_pair_history(value2[-7:], adv40, 3, op.ts_corr, 5)
    decay2 = op.ts_decay_linear(corr1, 5)[None, :]
    rank2 = op.rank(decay2)
    minimum1 = np.minimum(rank1, rank2)
    return op.clean(minimum1[0])


def alpha078(mf):
    """Alpha078：混合价格和/累计 adv40 相关排名，以 VWAP/成交量排名相关排名为幂。

    最少 63 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-63:] > 0
    low = np.where(active[-24:], mf['low'][-24:], np.nan)
    vwap = np.where(active[-24:], mf['vwap'][-24:], np.nan)
    sum1 = op.ts_history(low * 0.352233 + vwap * 0.647767, 19, op.ts_sum, 6)
    total_turnover = np.where(active[-63:], mf['total_turnover'][-63:], np.nan)
    adv40 = op.ts_history(total_turnover, 40, op.ts_mean, 24)
    sum2 = op.ts_history(adv40, 19, op.ts_sum, 6)
    corr1 = op.ts_corr(sum1, sum2, 6)[None, :]
    rank1 = op.rank(corr1)
    rank2 = op.rank(vwap[-5:])
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    rank3 = op.rank(volume)
    corr2 = op.ts_corr(rank2, rank3, 5)[None, :]
    rank4 = op.rank(corr2)
    return op.clean(op.power(rank1, rank4)[0])


def alpha079(mf):
    """Alpha079：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha080(mf):
    """Alpha080：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha081(mf):
    """Alpha081：流动性相关排名幂的滚动乘积对数排名，与 VWAP/量排名相关比较。

    最少 78 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-78:] > 0
    vwap = np.where(active[-21:], mf['vwap'][-21:], np.nan)
    total_turnover = np.where(active[-78:], mf['total_turnover'][-78:], np.nan)
    adv10 = op.ts_history(total_turnover, 10, op.ts_mean, 69)
    sum1 = op.ts_history(adv10, 49, op.ts_sum, 21)
    corr1 = op.ts_pair_history(vwap, sum1, 8, op.ts_corr, 14)
    rank1 = op.rank(corr1)
    rank2 = op.rank(op.power(rank1, 4))
    product1 = op.ts_product(rank2, 14)[None, :]
    log1 = op.log(product1)
    rank3 = op.rank(log1)
    rank4 = op.rank(vwap[-5:])
    volume = np.where(active[-5:], mf['volume'][-5:], np.nan)
    rank5 = op.rank(volume)
    corr2 = op.ts_corr(rank4, rank5, 5)[None, :]
    rank6 = op.rank(corr2)
    condition1 = op.compare(rank3, rank6, np.less)
    return op.clean((-condition1)[0])


def alpha082(mf):
    """Alpha082：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha083(mf):
    """Alpha083：延迟相对振幅排名与成交量排名，除以振幅/VWAP价差比。

    最少 7 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-7:] > 0
    high = np.where(active[-3:], mf['high'][-3:], np.nan)
    low = np.where(active[-3:], mf['low'][-3:], np.nan)
    close = np.where(active[-7:], mf['close'][-7:], np.nan)
    mean1 = op.ts_history(close, 5, op.ts_mean, 3)
    value2 = op.safe_divide(high - low, mean1)
    lag1 = value2[:-2]
    rank1 = op.rank(lag1)
    volume = np.where(active[-1:], mf['volume'][-1:], np.nan)
    rank2 = op.rank(volume)
    rank3 = op.rank(rank2)
    vwap = np.where(active[-1:], mf['vwap'][-1:], np.nan)
    return op.clean(op.safe_divide(rank1 * rank3, op.safe_divide(value2[-1:], vwap - close[-1:]))[0])


def alpha084(mf):
    """Alpha084：VWAP 距高点的时间排名，以收盘变化为带符号幂。

    最少 34 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-34:] > 0
    vwap = np.where(active[-34:], mf['vwap'][-34:], np.nan)
    max1 = op.ts_history(vwap, 15, op.ts_max, 20)
    ts_rank1 = op.ts_rank(vwap[-20:] - max1, 20)[None, :]
    close = np.where(active[-5:], mf['close'][-5:], np.nan)
    delta1 = op.delta(close, 4)[-1:]
    signed_power1 = op.signed_power(ts_rank1, delta1)
    return op.clean(signed_power1[0])


def alpha085(mf):
    """Alpha085：高收混合价/adv30 相关排名，以中间价/量时间排名相关排名为幂。

    最少 38 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-38:] > 0
    high = np.where(active[-9:], mf['high'][-9:], np.nan)
    close = np.where(active[-9:], mf['close'][-9:], np.nan)
    total_turnover = np.where(active[-38:], mf['total_turnover'][-38:], np.nan)
    adv30 = op.ts_history(total_turnover, 30, op.ts_mean, 9)
    corr1 = op.ts_corr(high * 0.876703 + close * 0.12329699999999999, adv30, 9)[None, :]
    rank1 = op.rank(corr1)
    low = np.where(active[-9:], mf['low'][-9:], np.nan)
    ts_rank1 = op.ts_history(op.safe_divide(high + low, 2), 3, op.ts_rank, 7)
    volume = np.where(active[-16:], mf['volume'][-16:], np.nan)
    ts_rank2 = op.ts_history(volume, 10, op.ts_rank, 7)
    corr2 = op.ts_corr(ts_rank1, ts_rank2, 7)[None, :]
    rank2 = op.rank(corr2)
    return op.clean(op.power(rank1, rank2)[0])


def alpha086(mf):
    """Alpha086：收盘/累计 adv20 相关的时间排名与收盘/VWAP 价差排名比较。

    最少 57 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-57:] > 0
    close = np.where(active[-25:], mf['close'][-25:], np.nan)
    total_turnover = np.where(active[-57:], mf['total_turnover'][-57:], np.nan)
    adv20 = op.ts_history(total_turnover, 20, op.ts_mean, 38)
    sum1 = op.ts_history(adv20, 14, op.ts_sum, 25)
    corr1 = op.ts_pair_history(close, sum1, 6, op.ts_corr, 20)
    ts_rank1 = op.ts_rank(corr1, 20)[None, :]
    vwap = np.where(active[-1:], mf['vwap'][-1:], np.nan)
    rank1 = op.rank(close[-1:] - vwap)
    condition1 = op.compare(ts_rank1, rank1, np.less)
    return op.clean((-condition1)[0])


def alpha087(mf):
    """Alpha087：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha088(mf):
    """Alpha088：OHLC 排名差的衰减排名，与收盘/adv60 时间排名相关信号取小。

    最少 92 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-92:] > 0
    open_ = np.where(active[-8:], mf['open'][-8:], np.nan)
    rank1 = op.rank(open_)
    low = np.where(active[-8:], mf['low'][-8:], np.nan)
    rank2 = op.rank(low)
    high = np.where(active[-8:], mf['high'][-8:], np.nan)
    rank3 = op.rank(high)
    close = np.where(active[-21:], mf['close'][-21:], np.nan)
    rank4 = op.rank(close[-8:])
    decay1 = op.ts_decay_linear(rank1 + rank2 - rank3 - rank4, 8)[None, :]
    rank5 = op.rank(decay1)
    ts_rank1 = op.ts_history(close, 8, op.ts_rank, 14)
    total_turnover = np.where(active[-92:], mf['total_turnover'][-92:], np.nan)
    adv60 = op.ts_history(total_turnover, 60, op.ts_mean, 33)
    ts_rank2 = op.ts_history(adv60, 20, op.ts_rank, 14)
    corr1 = op.ts_pair_history(ts_rank1, ts_rank2, 8, op.ts_corr, 7)
    decay2 = op.ts_history(corr1, 6, op.ts_decay_linear, 2)
    ts_rank3 = op.ts_rank(decay2, 2)[None, :]
    minimum1 = np.minimum(rank5, ts_rank3)
    return op.clean(minimum1[0])


def alpha089(mf):
    """Alpha089：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha090(mf):
    """Alpha090：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha091(mf):
    """Alpha091：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha092(mf):
    """Alpha092：价格位置条件与低价/adv30 排名相关性的衰减时间排名，取小。

    最少 46 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-46:] > 0
    high = np.where(active[-31:], mf['high'][-31:], np.nan)
    low = np.where(active[-31:], mf['low'][-31:], np.nan)
    close = np.where(active[-31:], mf['close'][-31:], np.nan)
    open_ = np.where(active[-31:], mf['open'][-31:], np.nan)
    condition1 = op.compare(op.safe_divide(high + low, 2) + close, low + open_, np.less)
    decay1 = op.ts_history(condition1, 14, op.ts_decay_linear, 18)
    ts_rank1 = op.ts_rank(decay1, 18)[None, :]
    rank1 = op.rank(low[-17:])
    total_turnover = np.where(active[-46:], mf['total_turnover'][-46:], np.nan)
    adv30 = op.ts_history(total_turnover, 30, op.ts_mean, 17)
    rank2 = op.rank(adv30)
    corr1 = op.ts_pair_history(rank1, rank2, 7, op.ts_corr, 11)
    decay2 = op.ts_history(corr1, 6, op.ts_decay_linear, 6)
    ts_rank2 = op.ts_rank(decay2, 6)[None, :]
    minimum1 = np.minimum(ts_rank1, ts_rank2)
    return op.clean(minimum1[0])


def alpha093(mf):
    """Alpha093：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha094(mf):
    """Alpha094：VWAP 距低点排名，以 VWAP/adv60 时间排名相关的时间排名为幂。

    最少 81 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-81:] > 0
    vwap = np.where(active[-37:], mf['vwap'][-37:], np.nan)
    min1 = op.ts_min(vwap[-11:], 11)[None, :]
    rank1 = op.rank(vwap[-1:] - min1)
    ts_rank1 = op.ts_history(vwap, 19, op.ts_rank, 19)
    total_turnover = np.where(active[-81:], mf['total_turnover'][-81:], np.nan)
    adv60 = op.ts_history(total_turnover, 60, op.ts_mean, 22)
    ts_rank2 = op.ts_history(adv60, 4, op.ts_rank, 19)
    corr1 = op.ts_pair_history(ts_rank1, ts_rank2, 18, op.ts_corr, 2)
    ts_rank3 = op.ts_rank(corr1, 2)[None, :]
    return op.clean((-op.power(rank1, ts_rank3))[0])


def alpha095(mf):
    """Alpha095：开盘距低点排名，与中间价和/累计 adv40 相关排名幂的时间排名比较。

    最少 79 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-79:] > 0
    open_ = np.where(active[-12:], mf['open'][-12:], np.nan)
    min1 = op.ts_min(open_, 12)[None, :]
    rank1 = op.rank(open_[-1:] - min1)
    high = np.where(active[-40:], mf['high'][-40:], np.nan)
    low = np.where(active[-40:], mf['low'][-40:], np.nan)
    sum1 = op.ts_history(op.safe_divide(high + low, 2), 19, op.ts_sum, 22)
    total_turnover = np.where(active[-79:], mf['total_turnover'][-79:], np.nan)
    adv40 = op.ts_history(total_turnover, 40, op.ts_mean, 40)
    sum2 = op.ts_history(adv40, 19, op.ts_sum, 22)
    corr1 = op.ts_pair_history(sum1, sum2, 12, op.ts_corr, 11)
    rank2 = op.rank(corr1)
    ts_rank1 = op.ts_rank(op.power(rank2, 5), 11)[None, :]
    condition1 = op.compare(rank1, ts_rank1, np.less)
    return op.clean(condition1[0])


def alpha096(mf):
    """Alpha096：VWAP/量排名相关信号，与收盘/adv60 排名相关峰值位置信号取大。

    最少 101 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-101:] > 0
    vwap = np.where(active[-13:], mf['vwap'][-13:], np.nan)
    rank1 = op.rank(vwap)
    volume = np.where(active[-13:], mf['volume'][-13:], np.nan)
    rank2 = op.rank(volume)
    corr1 = op.ts_pair_history(rank1, rank2, 3, op.ts_corr, 11)
    decay1 = op.ts_history(corr1, 4, op.ts_decay_linear, 8)
    ts_rank1 = op.ts_rank(decay1, 8)[None, :]
    close = np.where(active[-45:], mf['close'][-45:], np.nan)
    ts_rank2 = op.ts_history(close, 7, op.ts_rank, 39)
    total_turnover = np.where(active[-101:], mf['total_turnover'][-101:], np.nan)
    adv60 = op.ts_history(total_turnover, 60, op.ts_mean, 42)
    ts_rank3 = op.ts_history(adv60, 4, op.ts_rank, 39)
    corr2 = op.ts_pair_history(ts_rank2, ts_rank3, 3, op.ts_corr, 37)
    argmax1 = op.ts_history(corr2, 12, op.ts_argmax, 26)
    decay2 = op.ts_history(argmax1, 14, op.ts_decay_linear, 13)
    ts_rank4 = op.ts_rank(decay2, 13)[None, :]
    maximum1 = np.maximum(ts_rank1, ts_rank4)
    return op.clean((-maximum1)[0])


def alpha097(mf):
    """Alpha097：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha098(mf):
    """Alpha098：VWAP/累计 adv5 衰减相关排名，减开盘/adv15 排名相关谷值位置信号。

    最少 53 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-53:] > 0
    vwap = np.where(active[-10:], mf['vwap'][-10:], np.nan)
    total_turnover = np.where(active[-53:], mf['total_turnover'][-53:], np.nan)
    mean1 = op.ts_history(total_turnover[-39:], 5, op.ts_mean, 35)
    sum1 = op.ts_history(mean1, 26, op.ts_sum, 10)
    corr1 = op.ts_pair_history(vwap, sum1, 4, op.ts_corr, 7)
    decay1 = op.ts_decay_linear(corr1, 7)[None, :]
    rank1 = op.rank(decay1)
    open_ = np.where(active[-39:], mf['open'][-39:], np.nan)
    rank2 = op.rank(open_)
    adv15 = op.ts_history(total_turnover, 15, op.ts_mean, 39)
    rank3 = op.rank(adv15)
    corr2 = op.ts_pair_history(rank2, rank3, 20, op.ts_corr, 20)
    argmin1 = op.ts_history(corr2, 8, op.ts_argmin, 13)
    ts_rank1 = op.ts_history(argmin1, 6, op.ts_rank, 8)
    decay2 = op.ts_decay_linear(ts_rank1, 8)[None, :]
    rank4 = op.rank(decay2)
    return op.clean((rank1 - rank4)[0])


def alpha099(mf):
    """Alpha099：中间价和/累计 adv60 相关排名与低价/成交量相关排名比较。

    最少 85 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-85:] > 0
    high = np.where(active[-26:], mf['high'][-26:], np.nan)
    low = np.where(active[-26:], mf['low'][-26:], np.nan)
    sum1 = op.ts_history(op.safe_divide(high + low, 2), 19, op.ts_sum, 8)
    total_turnover = np.where(active[-85:], mf['total_turnover'][-85:], np.nan)
    adv60 = op.ts_history(total_turnover, 60, op.ts_mean, 26)
    sum2 = op.ts_history(adv60, 19, op.ts_sum, 8)
    corr1 = op.ts_corr(sum1, sum2, 8)[None, :]
    rank1 = op.rank(corr1)
    volume = np.where(active[-6:], mf['volume'][-6:], np.nan)
    corr2 = op.ts_corr(low[-6:], volume, 6)[None, :]
    rank2 = op.rank(corr2)
    condition1 = op.compare(rank1, rank2, np.less)
    return op.clean((-condition1)[0])


def alpha100(mf):
    """Alpha100：依赖行业数据，暂不计算，按资产位置返回 NaN。"""
    return np.full(len(mf.assets), np.nan)


def alpha101(mf):
    """Alpha101：日内实体涨跌除以振幅加 0.001。

    最少 1 日历史；输出顺序对应 mf.assets。
    """
    active = mf['volume'][-1:] > 0
    close = np.where(active[-1:], mf['close'][-1:], np.nan)
    open_ = np.where(active[-1:], mf['open'][-1:], np.nan)
    high = np.where(active[-1:], mf['high'][-1:], np.nan)
    low = np.where(active[-1:], mf['low'][-1:], np.nan)
    return op.clean(op.safe_divide(close - open_, high - low + 0.001)[0])


# 按因子分别预热；保留原公式历史上限，行业占位函数本身不读取历史。
FACTOR_LOOKBACKS = (
    24, 8, 10, 9, 10, 10, 67, 15, 6, 5,  # Alpha001～010
    4, 2, 5, 10, 5, 5, 24, 10, 250, 2,  # Alpha011～020
    20, 20, 20, 200, 20, 11, 7, 24, 11, 20,  # Alpha021～030
    31, 235, 1, 5, 32, 200, 201, 10, 250, 10,  # Alpha031～040
    1, 1, 39, 5, 25, 21, 20, 252, 21, 9,  # Alpha041～050
    21, 240, 10, 1, 17, 10, 31, 13, 26, 10,  # Alpha051～060
    196, 49, 239, 146, 72, 16, 25, 34, 31, 82,  # Alpha061～070
    225, 56, 20, 80, 61, 140, 46, 63, 171, 18,  # Alpha071～080
    78, 34, 7, 34, 38, 57, 109, 92, 27, 46,  # Alpha081～090
    34, 46, 121, 81, 79, 101, 115, 53, 85, 30,  # Alpha091～100
    1,  # Alpha101～101
)
HISTORY_DAYS = max(FACTOR_LOOKBACKS)
FACTORS = (
    alpha001, alpha002, alpha003, alpha004, alpha005, alpha006, alpha007, alpha008, alpha009, alpha010,
    alpha011, alpha012, alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha019, alpha020,
    alpha021, alpha022, alpha023, alpha024, alpha025, alpha026, alpha027, alpha028, alpha029, alpha030,
    alpha031, alpha032, alpha033, alpha034, alpha035, alpha036, alpha037, alpha038, alpha039, alpha040,
    alpha041, alpha042, alpha043, alpha044, alpha045, alpha046, alpha047, alpha048, alpha049, alpha050,
    alpha051, alpha052, alpha053, alpha054, alpha055, alpha056, alpha057, alpha058, alpha059, alpha060,
    alpha061, alpha062, alpha063, alpha064, alpha065, alpha066, alpha067, alpha068, alpha069, alpha070,
    alpha071, alpha072, alpha073, alpha074, alpha075, alpha076, alpha077, alpha078, alpha079, alpha080,
    alpha081, alpha082, alpha083, alpha084, alpha085, alpha086, alpha087, alpha088, alpha089, alpha090,
    alpha091, alpha092, alpha093, alpha094, alpha095, alpha096, alpha097, alpha098, alpha099, alpha100,
    alpha101,
)
