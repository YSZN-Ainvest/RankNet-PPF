"""逐日开盘记账。权重由调用方在主循环中生成。"""

from .book import PortfolioBook,np
import matplotlib.pyplot as plt

__all__ = ["PortfolioBook","show_book"]


def show_book(book):
    history = book.to_frame()
    if history.empty:
        raise ValueError("book 尚无回测记录")

    dates = history["date"]
    equity = history["equity"].to_numpy(dtype=float)
    returns = history["net_return"].to_numpy(dtype=float)
    fees = history["fee"].to_numpy(dtype=float)

    # 将初始资金纳入历史峰值，避免漏掉回测开始后的回撤。
    peaks = np.maximum.accumulate(
        np.r_[book.initial_cash, equity]
    )[1:]
    drawdown = equity / peaks - 1

    days = len(history)
    total_return = equity[-1] / book.initial_cash - 1
    annual_return = (equity[-1] / book.initial_cash) ** (252 / days) - 1
    volatility = np.std(returns, ddof=1) if days > 1 else np.nan
    sharpe = (
        np.mean(returns) / volatility * np.sqrt(252)
        if volatility > 0 else np.nan
    )

    print(f"回测交易日：{days}")
    print(f"最终净值：{equity[-1]:.4f}")
    print(f"累计收益：{total_return:.2%}")
    print(f"年化收益（按 252 天）：{annual_return:.2%}")
    print(f"年化波动率：{volatility * np.sqrt(252):.2%}")
    print(f"夏普比率（无风险利率按 0 计）：{sharpe:.2f}")
    print(f"最大回撤：{drawdown.min():.2%}")
    print(f"正收益日占比：{np.mean(returns > 0):.2%}")
    print(f"累计手续费：{fees.sum():.6f}")
    print(f"累计退市损益：{history['delisting_pnl'].sum():.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(dates, equity)
    axes[0, 0].set_title("Equity")

    axes[0, 1].fill_between(dates, drawdown, 0, alpha=0.5)
    axes[0, 1].set_title("Drawdown")

    axes[1, 0].hist(returns, bins=30)
    axes[1, 0].set_title("Daily net returns")

    axes[1, 1].plot(dates, fees)
    axes[1, 1].set_title("Daily fees")

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    plt.show()

    return history