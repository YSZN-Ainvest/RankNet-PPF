import numpy as np
import pandas as pd

from .book import PortfolioBook


def book_summary(history, initial_cash=1.0):
    equity = history['equity'].to_numpy()
    returns = history['net_return'].to_numpy()
    peaks = np.maximum.accumulate(np.r_[initial_cash, equity])[1:]
    volatility = np.std(returns, ddof=1)
    return {
        'days': len(history),
        'equity': equity[-1],
        'total_return': equity[-1] / initial_cash - 1,
        'annual_return': (equity[-1] / initial_cash) ** (252 / len(history)) - 1,
        'volatility': volatility * np.sqrt(252),
        'sharpe': np.mean(returns) / volatility * np.sqrt(252),
        'max_drawdown': (equity / peaks - 1).min(),
        'fee': history['fee'].sum(),
        'delisting_pnl': history['delisting_pnl'].sum(),
    }


def save_books(books, BACKTEST_PATH):
    BACKTEST_PATH.mkdir(parents=True, exist_ok=True)
    summary = []
    summary_by_year = []
    for name, group_books in books.items():
        for group, book in enumerate(group_books, 1):
            history = book.to_frame()
            history.to_parquet(BACKTEST_PATH / f'{name}_{group:02d}.parquet', index=False)
            summary.append({'factor': name, 'group': group, **book_summary(history)})
            initial_cash = book.initial_cash
            for year, year_history in history.groupby(history['date'].dt.year):
                summary_by_year.append({
                    'factor': name, 'group': group, 'year': year,
                    **book_summary(year_history, initial_cash),
                })
                initial_cash = year_history['equity'].iloc[-1]
    summary = pd.DataFrame(summary)
    summary.to_csv(BACKTEST_PATH / 'summary.csv', index=False)
    pd.DataFrame(summary_by_year).to_csv(BACKTEST_PATH / 'summary_by_year.csv', index=False)
    return summary
