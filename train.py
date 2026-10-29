
"""用 2020–2022 年日截面训练排序模型，逐轮报告 IC 并保存模型。"""

import pandas as pd
import numpy as np

import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader


from TOOL.model import ColTimeSample, TabularResNet, PairwiseSoftplusLoss

WORK_PATH = Path(__file__).resolve().parent
FEATURE_PATH = WORK_PATH / 'rank_output/factor_rank_20日'
LABEL_PATH = WORK_PATH / 'label_output/10日'
MODEL_PATH = WORK_PATH / 'PAIR'
TRAIN=['2020','2021','2022']
SEED = 42


random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


feature_paths = {p.stem: p for p in FEATURE_PATH.glob('*/*.parquet')}
label_paths = {p.stem: p for p in LABEL_PATH.glob('*/*.parquet')}
dates = sorted(feature_paths.keys() & label_paths.keys())

# 特征和标签按同日配对。
date_paths = pd.DataFrame({
    'date': pd.to_datetime(dates, format='%Y%m%d'),
    'feature_path': [feature_paths[date] for date in dates],
    'label_path': [label_paths[date] for date in dates],
})


def load_xy(date, label_func,df=False):
    """label_func 接收以股票代码为索引的 Y 表，返回同索引的一维 Series。"""
    row = date_paths.loc[date_paths['date'] == pd.Timestamp(date)].iloc[0]
    x = pd.read_parquet(row['feature_path']).set_index('order_book_id')
    y = pd.read_parquet(row['label_path']).set_index('order_book_id')

    x = x.drop(columns='datetime').fillna(0.5)
    if df:
        return y.drop(columns='datetime').fillna(0)
    y = label_func(y.drop(columns='datetime').fillna(0)).fillna(0)

    # 只保留双方都有的股票，并保证 X、Y 的顺序一致。
    assets = x.index.intersection(y.index, sort=False)
    return x.loc[assets].to_numpy(), y.loc[assets].to_numpy()


# 示例：以第一段开盘到开盘收益作为一维标签。
# x, y = load_xy('2020-02-07', lambda y: y['return_1'])

def calc_ic(date, label_func):
    """返回普通 IC 和秩 IC，形状均为 (1, 因子数)，列顺序与 X 一致。"""
    x, y = load_xy(date, label_func)
    x = pd.DataFrame(x)
    y = pd.Series(y)

    # 忽略零标准差产生的除法警告，无法计算的相关系数填为 0。
    with np.errstate(divide='ignore', invalid='ignore'):
        ic = x.corrwith(y).fillna(0).to_numpy()[None, :]
        rank_ic = x.rank().corrwith(y.rank()).fillna(0).to_numpy()[None, :]
    return ic, rank_ic


# ic, rank_ic = calc_ic('2020-02-07', lambda y: y['return_1'])


def averRank(df: pd.DataFrame, weight=None):
    return df.rank(axis=0, pct=True).mean(axis=1)


ds = ColTimeSample()
for date in date_paths['date']:
    if str(date.year) in TRAIN:
        ds.append(*load_xy(date, lambda y: averRank(y)))

# 一个样本是一整日的 (X, Y)；shuffle 只打乱日期组的顺序。
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
print(f'日期组数：{len(ds):,}；batch 数：{len(loader):,}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TabularResNet(input_dim=ds[0][0].shape[1], output_dim=1).to(device)
loss = PairwiseSoftplusLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_ics = []

    for x, y in loader:
        # batch_size=1，每次处理一整日的股票。
        x = x.squeeze(0).to(device=device, dtype=torch.float32)
        y = y.squeeze(0).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x).squeeze(-1)
        batch_loss = loss(pred, y)
        batch_loss.backward()
        optimizer.step()

        # 统计训练过程中各日期组的 Pearson IC，无法计算时记为 0。
        pred_values = pred.detach().cpu().numpy().reshape(-1)
        y_values = y.detach().cpu().numpy().reshape(-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ic = pd.Series(pred_values).corr(pd.Series(y_values))
        epoch_ics.append(0.0 if pd.isna(ic) else float(ic))

    ic_mean = np.mean(epoch_ics)
    ic_var = np.var(epoch_ics, ddof=0)
    ic_std = np.sqrt(ic_var)
    icir = ic_mean / ic_std if ic_std > 0 else 0.0  # 不年化。
    print(f'Epoch {epoch + 1}/{num_epochs}, IC均值={ic_mean:.6f}, IC方差={ic_var:.6f}, ICIR={icir:.6f}')


torch.save(model, MODEL_PATH)
