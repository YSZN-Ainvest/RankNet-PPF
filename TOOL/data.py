"""行情与实验结果的路径配置，以及逐日研究循环共用的数据读写。"""

from pathlib import Path

import pandas as pd


WORK_PATH = Path(__file__).resolve().parents[1]
DAILY_DATA_PATH = Path('/mnt/nas/data/whole_history')
OUTPUT_PATH = WORK_PATH / 'output'
RANK_PATH = WORK_PATH / 'rank_output' / 'factor_rank_20日'
MODEL_PATH = WORK_PATH / 'PAIR'
MODEL_OUTPUT_PATH = WORK_PATH / 'model_output'
BACKTEST_PATH = MODEL_OUTPUT_PATH / 'backtest'


def preload(years_list):
    """预加载股票资料与排好序的交易日期、行情地址；不预读全部行情。"""
    base_info_path = DAILY_DATA_PATH / 'instruments' / 'stock.csv'
    daily_bar_path = DAILY_DATA_PATH / 'daily_bar'
    base_info = pd.read_csv(base_info_path)
    dates_path_list = sorted(
        path for year in years_list
        for path in (daily_bar_path / year).glob('*.csv')
    )
    dates_list = [f'{p.stem[:4]}-{p.stem[4:6]}-{p.stem[6:8]}' for p in dates_path_list]
    return base_info, dates_list, dates_path_list


def load_market(date, date_path, base_info, fields):
    """每天循环中共用的行情读取和在市股票筛选。"""
    df = pd.read_csv(date_path)
    listed_asset = base_info[
        (base_info['type'] == 'CS') &
        (base_info['listed_date'] <= date) &
        ((base_info['de_listed_date'] > date) |
         (base_info['de_listed_date'] == '0000-00-00')) &
        (base_info['exchange'] != 'BJSE')
    ]['order_book_id']
    listed_info = df[df['order_book_id'].isin(listed_asset)][
        ['order_book_id', 'datetime', *fields]
    ].copy()
    listed_info['datetime'] = date
    return listed_info


def date_path(path, date):
    return path / date[:4] / f"{date.replace('-', '')}.parquet"


def save_frame(frame, path, date):
    year_path = path / date[:4]
    year_path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(date_path(path, date), compression='zstd', index=False)


def save_scores(frame, path, date):
    frame = frame.rename_axis('order_book_id').reset_index().assign(datetime=date)
    save_frame(frame, path, date)
