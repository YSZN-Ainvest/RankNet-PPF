# 算子约定：输入为二维 ndarray，行=时间（由早到晚），列=资产。
# 两个输入应具有相同坐标与形状；由调用方保证 days 是正整数，
# delay/delta 另允许 days=0。算子不自动对齐坐标，也不屏蔽停牌。
#
# 输出：基础变换保留输入形状；ts_* 仅返回最新窗口的一维资产向量。
# rank_1d 是二维排名的内部基础，也可单独用于一维数据。
# 时序统计要求窗口完整有效；不以跳过 NaN 的方式缩短历史窗口。
# 本文件部分启用 njit
from .config import np,pd,njit



#=============================================================================
# 非有限值处理：使用 Numba 编译逐元素计算。
@njit
def clean(values):
    """一次遍历完成有限值检查和写入；支持二维输入及内部归约的一维结果。"""
    result = np.empty(values.shape)
    if values.ndim == 1:
        for i in range(values.shape[0]):
            value = values[i]
            result[i] = value if np.isfinite(value) else np.nan
    else:
        for t in range(values.shape[0]):
            for a in range(values.shape[1]):
                value = values[t, a]
                result[t, a] = value if np.isfinite(value) else np.nan
    return result



#=============================================================================
# 算术算子：逐元素除法与无效值处理。
@njit
def divide(numerator, denominator):
    """融合有效性检查与除法，直接访问原数组，不创建展平副本或布尔筛选数组。"""
    result = np.full(numerator.shape, np.nan)
    for t in range(numerator.shape[0]):
        for a in range(numerator.shape[1]):
            x = numerator[t, a]
            y = denominator[t, a]
            if np.isfinite(x) and np.isfinite(y) and y != 0:
                value = x / y
                if np.isfinite(value):
                    result[t, a] = value
    return result



#=============================================================================
# 秩算子，对于高维资产排序情况，适合原生numpy
def rank_1d(values, pct=True):
    """一维排名：跳过 NaN，并列取平均名次；无穷值按大小参与排序。

    pct=True 返回名次/有效数量，唯一有效值为 0.5；
    pct=False 返回从 1 开始的名次，唯一有效值为 1。
    """
    result = np.full(values.shape, np.nan)
    valid = ~np.isnan(values)
    sequence = values[valid]
    count = sequence.size
    if count == 0:
        return result
    # unique 按值排序；groups 把排序后的组映射回原序列的位置。
    # 累计数量是每组的末名次，减去半个组跨度得到平均名次。
    _, groups, counts = np.unique(sequence, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2.0
    if pct:
        average_ranks = average_ranks / count if count > 1 else np.array([0.5])
    result[valid] = average_ranks[groups]
    return result


def rank_array(values, axis=1, pct=True):
    """二维排名：axis=1 逐日对资产排名；axis=0 转置后逐资产对时间排名。

    输出与输入同形状。支持对应负轴 -1、-2；调用方保证 axis 有效。
    """
    # 统一逐行调用一维函数；时间轴排名通过转置复用同一流程。
    transpose = axis == 0 or axis == -2
    data = values.T if transpose else values
    result = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
        result[i] = rank_1d(data[i], pct=pct)
    return result.T if transpose else result


def rank(values):
    """逐日沿资产轴做横截面分位排名，返回时间 × 资产。"""
    # 通用 rank_array 只排除 NaN；行情 rank 先排除无穷值再排名。
    return rank_array(clean(values), axis=1, pct=True)



#=============================================================================
# 一些时序算子
# 延迟算子，纯拷贝，适合原生numpy
def delay(values, days):
    """沿时间轴向后移动 days 日，开头补 NaN，不跳过缺失日。"""
    # days 以输入的时间行数计数；输入没有的交易日无法在这里补回。
    result = np.full(values.shape, np.nan)
    if days == 0:
        result[:] = clean(values)
    elif days < len(values):
        result[days:] = clean(values[:-days])
    return result


# 时间差分：计算当前值与 days 日前数值之差。
@njit
def delta(values, days=1):
    """直接读取两端并计算差分，避免构造完整 delay 及多份 clean 中间数组。"""
    result = np.full(values.shape, np.nan)
    for t in range(days, values.shape[0]):
        for a in range(values.shape[1]):
            now = float(values[t, a])
            previous = float(values[t - days, a])
            if np.isfinite(now) and np.isfinite(previous):
                value = now - previous
                if np.isfinite(value):
                    result[t, a] = value
    return result



#=============================================================================
# 序列算子基座
def ts_apply(values, days, reducer):
    """截取最新完整窗口，调用归约函数，再统一屏蔽无效资产。

    reducer 接收 时间×资产 的窗口，返回一维资产向量。
    固定参数与结果索引由各归约函数负责；这里不分派算子类型。
    """
    # 历史不足时直接返回，不调用 reducer；只处理最后 days 行。
    if len(values) < days:
        return np.full(values.shape[1], np.nan)
    window = clean(values[-days:])
    valid = np.all(np.isfinite(window), axis=0)
    # reducer 会收到整个窗口，包括无效资产列；归约后统一屏蔽这些列。
    # 自定义 reducer 不应修改窗口，且必须返回长度为资产数的一维结果。
    return clean(np.where(valid, reducer(window), np.nan))



#=============================================================================
# 归约函数只负责计算；窗口长度与缺失规则由序列算子基座 ts_apply 统一处理。
# 将 axis、ddof、索引算子等固定细节保留在规约函数内。
# 归约函数应对infinite情况具有稳健性：对异常输入保证不抛出异常，无效计算结果在基座中被屏蔽。
def sum_reduce(window):
    """沿时间轴求和。"""
    return np.sum(window, axis=0)


def min_reduce(window):
    """沿时间轴取最小值。"""
    return np.min(window, axis=0)


def max_reduce(window):
    """沿时间轴取最大值。"""
    return np.max(window, axis=0)


@njit
def mean_reduce(window):
    """沿时间轴求均值；Numba 支持 sum 的 axis 参数。"""
    return np.sum(window, axis=0) / window.shape[0]


@njit
def stddev_reduce(window):
    """样本标准差；显式 ddof=1，避开 Numba 不支持的 std 参数组合。"""
    if len(window) == 1:
        return np.full(window.shape[1], np.nan)
    centered = window - mean_reduce(window)
    return np.sqrt(np.sum(centered * centered, axis=0) / (len(window) - 1))

@njit
def argmax_reduce(window):
    """最大值的位置：最早为 1，并列取最早。"""
    return np.argmax(window, axis=0) + 1



#=============================================================================
# 公开时序入口：二维历史 -> 最新资产向量，不生成历史滚动结果。
# 若后续还要对该结果做时序计算，需要调用方保存每日结果组成历史。
def ts_mean(values, days):
    """最新完整窗口的均值，返回资产向量。"""
    return ts_apply(values, days, mean_reduce)


def ts_sum(values, days):
    """最新完整窗口的和，返回资产向量。"""
    return ts_apply(values, days, sum_reduce)


def ts_min(values, days):
    """最新完整窗口的最小值，返回资产向量。"""
    return ts_apply(values, days, min_reduce)


def ts_max(values, days):
    """最新完整窗口的最大值，返回资产向量。"""
    return ts_apply(values, days, max_reduce)


def ts_stddev(values, days):
    """最新完整窗口的样本标准差，返回资产向量；单日窗口为空。"""
    return ts_apply(values, days, stddev_reduce)


def ts_argmax(values, days):
    """最新完整窗口中最大值的位置，返回资产向量。"""
    return ts_apply(values, days, argmax_reduce)


# 时间序列排名：计算最新值在窗口内的名次。
@njit
def ts_rank(values, days):
    """仅统计最新值的名次，不对窗口排序；完整窗口、并列平均、单值 0.5。"""
    result = np.full(values.shape[1], np.nan)
    if len(values) < days:
        return result
    for a in range(values.shape[1]):
        latest = values[-1, a]
        less = 0
        equal = 0
        valid = True
        for t in range(len(values) - days, len(values)):
            value = values[t, a]
            if not np.isfinite(value):
                valid = False
                break
            if value < latest:
                less += 1
            elif value == latest:
                equal += 1
        if valid:
            result[a] = 0.5 if days == 1 else (less + (equal + 1) / 2.0) / days
    return result


# 时间序列相关性：计算最新完整窗口的相关系数。
@njit
def ts_corr(left, right, days):
    """融合窗口检查、中心化与缩放；使用标量累计量，不构造中间矩阵。"""
    result = np.full(left.shape[1], np.nan)
    if len(left) < days or len(right) < days or days == 1:
        return result
    for a in range(left.shape[1]):
        total_x = 0.0
        total_y = 0.0
        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf
        valid = True
        for i in range(days):
            x = left[len(left) - days + i, a]
            y = right[len(right) - days + i, a]
            if not np.isfinite(x) or not np.isfinite(y):
                valid = False
                break
            total_x += x
            total_y += y
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
        if not valid or min_x == max_x or min_y == max_y:
            continue
        mean_x, mean_y = total_x / days, total_y / days
        scale_x, scale_y = 0.0, 0.0
        for i in range(days):
            scale_x = max(scale_x, abs(left[len(left) - days + i, a] - mean_x))
            scale_y = max(scale_y, abs(right[len(right) - days + i, a] - mean_y))
        if not np.isfinite(scale_x) or not np.isfinite(scale_y) or scale_x == 0 or scale_y == 0:
            continue
        cross, square_x, square_y = 0.0, 0.0, 0.0
        for i in range(days):
            x = (left[len(left) - days + i, a] - mean_x) / scale_x
            y = (right[len(right) - days + i, a] - mean_y) / scale_y
            cross += x * y
            square_x += x * x
            square_y += y * y
        value = cross / np.sqrt(square_x * square_y)
        if np.isfinite(value):
            result[a] = min(1.0, max(-1.0, value))
    return result














#=============================================================================
# Alpha101 补充算子。仍由调用方保证二维输入、坐标一致、days/periods 为正整数。
# abs/log/sign、逐元素 min/max 等直接使用 NumPy；不为已有 ufunc 再套一层。


def signed_power(values, exponent):
    """保留符号的幂：sign(x) * abs(x)**exponent；指数可为标量或同形数组。

    支持一维资产结果和二维历史；无效计算结果统一为 NaN。
    """
    return clean(np.sign(values) * np.abs(values) ** exponent)


@njit
def scale(values, target=1.0):
    """逐日缩放，使有效资产绝对值之和为 target；全零截面保留 NaN。

    输入输出为时间×资产；NaN/Inf 不参与分母计算。
    """
    result = np.full(values.shape, np.nan)
    for t in range(values.shape[0]):
        total = 0.0
        for a in range(values.shape[1]):
            if np.isfinite(values[t, a]):
                total += abs(values[t, a])
        if total > 0 and np.isfinite(total):
            for a in range(values.shape[1]):
                if np.isfinite(values[t, a]):
                    value = values[t, a] / total * target
                    if np.isfinite(value):
                        result[t, a] = value
    return result


@njit
def ts_product(values, days):
    """最新完整窗口的乘积；允许负数和零，无效值/溢出结果保留 NaN。"""
    result = np.full(values.shape[1], np.nan)
    if len(values) < days:
        return result
    result[:] = 1.0
    for t in range(len(values) - days, len(values)):
        for a in range(values.shape[1]):
            value = values[t, a]
            if np.isfinite(value):
                result[a] *= value
            else:
                result[a] = np.nan
    for a in range(values.shape[1]):
        if not np.isfinite(result[a]):
            result[a] = np.nan
    return result


@njit
def ts_argmin(values, days):
    """最新完整窗口的最小值位置；最早为 1，并列取最早。"""
    result = np.full(values.shape[1], np.nan)
    if len(values) < days:
        return result
    for a in range(values.shape[1]):
        minimum = np.inf
        position = 1
        valid = True
        for i in range(days):
            value = values[len(values) - days + i, a]
            if not np.isfinite(value):
                valid = False
                break
            if value < minimum:
                minimum = value
                position = i + 1
        if valid:
            result[a] = position
    return result


@njit
def ts_decay_linear(values, days):
    """最新完整窗口的线性加权均值；最早到最新权重为 1..days。"""
    result = np.full(values.shape[1], np.nan)
    if len(values) < days:
        return result
    result[:] = 0.0
    total_weight = days * (days + 1) / 2.0
    for i in range(days):
        weight = (i + 1) / total_weight
        for a in range(values.shape[1]):
            value = values[len(values) - days + i, a]
            if np.isfinite(value):
                result[a] += value * weight
            else:
                result[a] = np.nan
    for a in range(values.shape[1]):
        if not np.isfinite(result[a]):
            result[a] = np.nan
    return result


@njit
def ts_cov(values, other, days):
    """最新完整配对窗口的样本协方差，ddof=1；单日窗口为空。"""
    result = np.full(values.shape[1], np.nan)
    if len(values) < days or len(other) < days or days == 1:
        return result
    for a in range(values.shape[1]):
        total_x, total_y = 0.0, 0.0
        valid = True
        for i in range(days):
            x = values[len(values) - days + i, a]
            y = other[len(other) - days + i, a]
            if not np.isfinite(x) or not np.isfinite(y):
                valid = False
                break
            total_x += x
            total_y += y
        if not valid:
            continue
        mean_x, mean_y = total_x / days, total_y / days
        cross = 0.0
        for i in range(days):
            x = values[len(values) - days + i, a] - mean_x
            y = other[len(other) - days + i, a] - mean_y
            cross += x * y
        value = cross / (days - 1)
        if np.isfinite(value):
            result[a] = value
    return result


@njit
def indneutralize(values, groups):
    """逐日行业去均值；groups 为同形编码数组，未知为 -1/NaN。

    支持 MarketField 的浮点编码或整数数组；有效编码由入口保证为 0..G-1。
    缺失资产不进入行业均值，单资产行业为 0；历史行业允许变化。
    """
    result = np.full(values.shape, np.nan)
    group_count = 0
    for t in range(groups.shape[0]):
        for a in range(groups.shape[1]):
            code = groups[t, a]
            if np.isfinite(code) and code >= 0:
                group_count = max(group_count, int(code) + 1)
    sums = np.zeros(group_count)
    counts = np.zeros(group_count, dtype=np.int64)
    for t in range(values.shape[0]):
        sums[:] = 0.0
        counts[:] = 0
        for a in range(values.shape[1]):
            code = groups[t, a]
            if np.isfinite(code) and code >= 0 and np.isfinite(values[t, a]):
                group = int(code)
                sums[group] += values[t, a]
                counts[group] += 1
        for a in range(values.shape[1]):
            code = groups[t, a]
            if np.isfinite(code) and code >= 0 and np.isfinite(values[t, a]):
                group = int(code)
                value = values[t, a] - sums[group] / counts[group]
                if np.isfinite(value):
                    result[t, a] = value
    return result


def compare(left, right, comparison):
    """保留缺失值的比较，返回 0/1/NaN；支持 NumPy 广播。

    comparison 使用 np.less/greater/equal 等，也可为 np.logical_or/and。
    例如 compare(x, 0, np.less)；直接 x < 0 会将 NaN 静默变成 False。
    任一操作数无效则结果为空，逻辑运算也遵循这一完整有效规则。
    """
    return np.where(np.isfinite(left) & np.isfinite(right), comparison(left, right), np.nan)


@njit
def select(condition, yes, no):
    """保留空条件的三元选择；只要求被选分支有效。

    condition 为一维/二维 0/1/NaN 数组；yes/no 可广播到 condition 的形状。
    atleast_2d、broadcast_to 和最终 reshape 均为视图，不复制输入。
    统一二维遍历，将条件检查、选择和输出清理融合为一次循环。
    """
    conditions = np.atleast_2d(condition)
    positive = np.broadcast_to(np.atleast_2d(np.asarray(yes)), conditions.shape)
    negative = np.broadcast_to(np.atleast_2d(np.asarray(no)), conditions.shape)
    result = np.full(conditions.shape, np.nan)
    for t in range(result.shape[0]):
        for a in range(result.shape[1]):
            if np.isfinite(conditions[t, a]):
                value = positive[t, a] if conditions[t, a] != 0 else negative[t, a]
                if np.isfinite(value):
                    result[t, a] = value
    return result.reshape(condition.shape)


def ts_history(values, days, operator, periods):
    """计算最近 periods 个日期各自的时序结果，返回 periods×资产。

    operator 接收 (二维数组, days)，如 ts_stddev/ts_rank/ts_decay_linear。
    values[:end] 只是视图，不复制行情；逐行写入一个结果矩阵，避免 vstack 副本。
    早于输入起点的日期或不完整窗口输出 NaN；不缓存、不修改输入。
    """
    result = np.full((periods, values.shape[1]), np.nan)
    for i in range(periods):
        end = len(values) - periods + i + 1
        if end >= days:
            result[i] = operator(values[:end], days)
    return result


def ts_pair_history(left, right, days, operator, periods):
    """双输入版本的 ts_history；operator 可为 ts_corr/ts_cov。

    两个输入必须有相同坐标、长度；切片为视图，结果按时间从早到晚排列。
    """
    result = np.full((periods, left.shape[1]), np.nan)
    for i in range(periods):
        end = len(left) - periods + i + 1
        if end >= days:
            result[i] = operator(left[:end], right[:end], days)
    return result


#=============================================================================
# Alpha101 使用的数学变换与价格基准处理；调用方提供对齐的行情。
def safe_divide(numerator, denominator):
    """广播标量/二维数组后复用 divide；广播为视图，不复制行情。"""
    left, right = np.atleast_2d(numerator), np.atleast_2d(denominator)
    shape = np.broadcast_shapes(left.shape, right.shape)
    return divide(np.broadcast_to(left, shape), np.broadcast_to(right, shape))


@njit
def log(values):
    """二维逐元素自然对数；非正数、缺失及无穷值保留 NaN。"""
    result = np.full(values.shape, np.nan)
    for t in range(values.shape[0]):
        for a in range(values.shape[1]):
            value = values[t, a]
            if np.isfinite(value) and value > 0:
                result[t, a] = np.log(value)
    return result


def power(values, exponent):
    """普通实数幂，区别于 signed_power；无定义或非有限结果为 NaN。"""
    base, exp = np.broadcast_arrays(np.atleast_2d(values), np.atleast_2d(exponent))
    valid = (np.isfinite(base) & np.isfinite(exp)
             & ((base != 0) | (exp >= 0))
             & ((base >= 0) | (exp == np.floor(exp))))
    result = np.full(base.shape, np.nan)
    np.power(base, exp, out=result, where=valid)
    return clean(result)


@njit
def price_adjustment(close, prev_close):
    """以资产最后有效日为基准，反向连接相邻日期的除权/息价格比例。

    factor[t-1] = factor[t] * prev_close[t] / close[t-1]。
    返回与行情同形的乘数，适用于 OHLC、prev_close、VWAP，不改变输入。
    保留停牌行参与连接；中间缺行导致无法连接时，更早的价格保持 NaN。
    已退市资产仍以其最后有效日为基准保留过去行情，供历史截面排名使用。
    """
    result = np.full(close.shape, np.nan)
    for a in range(close.shape[1]):
        factor = 1.0
        for t in range(len(close) - 1, -1, -1):
            current = close[t, a]
            if not np.isfinite(current) or current <= 0:
                continue
            if np.isfinite(factor):
                result[t, a] = factor
            if t > 0:
                previous = close[t - 1, a]
                reference = prev_close[t, a]
                if np.isfinite(previous) and previous > 0 and np.isfinite(reference) and reference > 0:
                    factor *= reference / previous
                else:
                    factor = np.nan
    return result
