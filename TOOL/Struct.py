# 数据流：每日行情 -> MarketQueue.to_frame() 长表 -> MarketField 规则张量。
# MarketField['close'] 等字段切片为 时间×资产 的二维 ndarray，可交给算子。
from .config import np,pd
from .config import deque

#=============================================================================
class MarketQueue:
    """最多保留 max_days 天；调用方按交易日顺序，每天入队一次。"""

    def __init__(self, max_days=25):
        # 两个有界队列同步淘汰最早一天；max_days 由调用方保证为正整数。
        self.frames = deque(maxlen=max_days)
        self.dates = deque(maxlen=max_days)

    def append(self, date, market):
        """复制当天行情并记录日期，后续计算不修改调用方的数据。"""
        # 日行情统一使用当天零点；行情副本与日期一起入队。
        day = pd.Timestamp(date).normalize()
        frame = market.copy()
        frame['datetime'] = day
        self.frames.append(frame)
        self.dates.append(day)

    def to_frame(self):
        """一次拼接队列中的所有日行情。"""
        # 调用前至少入队一次；拼接后原始行索引不再保留。
        # 本方法不传递 self.dates：整日无记录的日期不会出现在返回表中。
        return pd.concat(self.frames, ignore_index=True)




#=============================================================================
# 输入约定：日期类型一致，日期/资产非空，同一日期与资产至多一行；
# 除两个标识列外，其余字段均可转换为浮点数。此处不重复校验输入。
# 日期和资产轴取表中实际出现值的并集；不会推断完整交易日历或上市状态。
class MarketField:
    def __init__(self, frame):
        # 坐标轴排序；字段轴保留输入列顺序，供下方批量赋值和字段查询。
        self.assets = np.sort(frame["order_book_id"].unique())
        self.dates = np.sort(frame["datetime"].unique())
        self.fields = frame.columns.drop(["order_book_id", "datetime"])

        # 维度顺序：日期 × 资产 × 字段；缺失的日期/资产组合保持 NaN。
        # 停牌记录的原始数值仍会写入，不在数据容器中自动屏蔽零成交量。
        self.tensor = np.full((len(self.dates), len(self.assets), len(self.fields)), np.nan)

        # 每条记录对应的资产位置、日期位置
        asset_idx = np.searchsorted(self.assets, frame["order_book_id"].to_numpy())
        date_idx = np.searchsorted(self.dates, frame["datetime"].to_numpy())

        self.tensor[date_idx, asset_idx, :] = frame[self.fields].to_numpy(
            dtype=float, na_value=np.nan
        )        
    def __getitem__(self, key):
        # 返回与 tensor 共享内存的二维视图；修改它会同时修改原张量。
        # 股票顺序对应 self.assets，日期顺序对应 self.dates。
        return self.tensor[:, :, self.fields.get_loc(key)]
