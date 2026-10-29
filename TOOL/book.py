"""接收当日行情和目标权重，在当日开盘立即再平衡。"""

import numpy as np
import pandas as pd


class PortfolioBook:
    """记录一个组合的持仓、现金和逐日收益。

    step(market, weights) 立即按 market 的开盘价执行 weights。
    weights 是以 order_book_id 为索引的 Series，数值为交易后的目标净值权重。
    允许多头、空头和现金；不会替调用方改变权重或延后交易。
    weights=None 表示保持现有持仓，空 Series 表示全部平仓。
    旧持仓若从当日行情中消失，视为退市并以零残值移除，不计成交或手续费。
    """

    def __init__(self, initial_cash=1.0, fee_per_side=0.00025):
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("初始资金必须为正")
        if not np.isfinite(fee_per_side) or fee_per_side < 0:
            raise ValueError("单边费率必须为有限非负数")

        self.initial_cash = float(initial_cash)
        self.fee_per_side = float(fee_per_side)
        self.cash = float(initial_cash)
        self.equity = float(initial_cash)
        self._holdings = pd.Series(dtype=float)
        self._last_quotes = None
        self._last_date = None
        self._records = []

    @property
    def holdings(self):
        """最近一次调仓后的股票持仓金额，正数为多头，负数为空头。"""
        return self._holdings.copy()

    def _read_market(self, market):
        required = ["datetime", "order_book_id", "open", "close", "prev_close"]
        if not isinstance(market, pd.DataFrame) or market.empty:
            raise ValueError("当日行情必须是非空 DataFrame")
        if not set(required).issubset(market.columns):
            raise ValueError(f"当日行情需要字段：{required}")
        if market["order_book_id"].isna().any() or market["order_book_id"].duplicated().any():
            raise ValueError("当日行情的股票代码不能为空或重复")

        dates = pd.to_datetime(market["datetime"]).dropna().unique()
        if len(dates) != 1:
            raise ValueError("每次只能传入一个交易日的行情")
        date = pd.Timestamp(dates[0])
        if self._last_date is not None and date <= self._last_date:
            raise ValueError("交易日必须严格递增")

        quotes = market.set_index("order_book_id")[["open", "close", "prev_close"]].copy()
        for column in quotes:
            quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
        return date, quotes

    def _mark_holdings(self, quotes, date):
        """估值仍在行情中的旧仓；消失的旧仓按零残值减记。"""
        if self._holdings.empty:
            return self._holdings.copy(), 0.0, []

        previous = self._last_quotes.reindex(self._holdings.index)
        disappeared = self._holdings.index.difference(quotes.index, sort=False)
        remaining = self._holdings.index.intersection(quotes.index, sort=False)
        previous_remaining = previous.reindex(remaining)
        current = quotes.reindex(remaining)
        valid = (
            np.isfinite(previous_remaining["open"]) & (previous_remaining["open"] > 0)
            & np.isfinite(previous_remaining["close"]) & (previous_remaining["close"] > 0)
            & np.isfinite(current["open"]) & (current["open"] > 0)
            & np.isfinite(current["prev_close"]) & (current["prev_close"] > 0)
        )
        if not valid.all():
            missing = remaining[~valid].tolist()[:5]
            raise ValueError(f"{date.date()} 旧持仓无法按开盘价估值：{missing}")

        growth = (
            previous_remaining["close"] / previous_remaining["open"]
            * current["open"] / current["prev_close"]
        )
        marked = self._holdings.reindex(remaining) * growth

        previous_disappeared = previous.reindex(disappeared)
        valid_close = (
            np.isfinite(previous_disappeared["open"])
            & (previous_disappeared["open"] > 0)
            & np.isfinite(previous_disappeared["close"])
            & (previous_disappeared["close"] > 0)
        )
        if not valid_close.all():
            missing = disappeared[~valid_close].tolist()[:5]
            raise ValueError(f"{date.date()} 退市持仓缺少最后收盘价：{missing}")
        last_close_value = self._holdings.reindex(disappeared) * (
            previous_disappeared["close"] / previous_disappeared["open"]
        )
        delisting_pnl = -float(last_close_value.sum())
        return marked, delisting_pnl, disappeared.tolist()

    def _read_weights(self, weights, quotes, date):
        if weights is None:
            return None
        if not isinstance(weights, pd.Series):
            raise TypeError("目标权重必须是以 order_book_id 为索引的 Series")
        if weights.index.has_duplicates or weights.index.hasnans:
            raise ValueError("目标权重的股票代码不能为空或重复")

        target = pd.to_numeric(weights, errors="coerce").astype(float)
        if not np.isfinite(target.to_numpy()).all():
            raise ValueError("目标权重必须为有限数值")
        target = target[target != 0]
        current_open = quotes["open"].reindex(target.index)
        valid = np.isfinite(current_open) & (current_open > 0)
        if not valid.all():
            missing = target.index[~valid].tolist()[:5]
            raise ValueError(f"{date.date()} 目标股票无法按开盘价成交：{missing}")
        if self.fee_per_side * target.abs().sum() >= 1:
            raise ValueError("目标总敞口过大，无法收敛手续费记账")
        return target

    def _trade(self, marked, target, equity_before):
        """按扣费后净值确定目标持仓，再计算实际买卖金额。"""
        assets = marked.index.union(target.index, sort=False)
        old = marked.reindex(assets, fill_value=0).to_numpy(dtype=float)
        weights = target.reindex(assets, fill_value=0).to_numpy(dtype=float)

        equity_after = equity_before
        for _ in range(50):
            traded = np.abs(equity_after * weights - old).sum()
            updated = equity_before - self.fee_per_side * traded
            if updated <= 0:
                raise ValueError("手续费耗尽组合净值")
            if abs(updated - equity_after) <= 1e-13 * max(1.0, equity_before):
                equity_after = updated
                break
            equity_after = updated
        else:
            raise ArithmeticError("手续费记账未收敛")

        new_holdings = target * equity_after
        traded = float(np.abs(equity_after * weights - old).sum())
        fee = self.fee_per_side * traded
        return equity_after, new_holdings, traded, fee

    def step(self, market, weights):
        """以当日开盘价立即执行权重，并返回本次逐日记录。"""
        if self.equity == 0:
            raise ValueError("组合净值已归零，无法继续回测")
        date, quotes = self._read_market(market)
        marked, delisting_pnl, delisted_codes = self._mark_holdings(quotes, date)
        target = self._read_weights(weights, quotes, date)
        equity_before = float(self.cash + marked.sum())
        if not np.isfinite(equity_before) or equity_before < 0:
            raise ValueError("交易前组合净值非正")
        if equity_before == 0 and target is not None and not target.empty:
            raise ValueError("组合净值已归零，无法建立新持仓")

        if target is None:
            equity_after = equity_before
            new_holdings = marked
            traded = fee = 0.0
        elif equity_before == 0:
            equity_after = 0.0
            new_holdings = target
            traded = fee = 0.0
        else:
            equity_after, new_holdings, traded, fee = self._trade(
                marked, target, equity_before
            )

        record = {
            "date": date,
            "gross_return": equity_before / self.equity - 1,
            "net_return": equity_after / self.equity - 1,
            "equity_before_trade": equity_before,
            "equity": equity_after,
            "traded_notional": traded,
            "fee": fee,
            "delisting_pnl": delisting_pnl,
            "delisted_codes": ",".join(delisted_codes),
            "net_exposure": float(new_holdings.sum() / equity_after) if equity_after else 0.0,
            "gross_exposure": float(new_holdings.abs().sum() / equity_after) if equity_after else 0.0,
            "positions": len(new_holdings),
        }

        self.cash = float(equity_after - new_holdings.sum())
        self.equity = float(equity_after)
        self._holdings = new_holdings.copy()
        self._last_quotes = quotes
        self._last_date = date
        self._records.append(record)
        return record.copy()

    def to_frame(self):
        """逐日记录；不读文件，也不自动绘图或保存。"""
        return pd.DataFrame(self._records).copy()
