"""按日股票截面的数据集、残差排序网络、成对损失和模型加载。"""

import __main__

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ResidualBlock(nn.Module):
    """两层全连接变换与恒等分支相加，再经 GELU 激活。"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    """逐股处理因子向量，输出用于横截面排序的单个分数。"""
    def __init__(self, input_dim=50, hidden_dim=256, num_blocks=4, output_dim=1, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


def load_model(MODEL_PATH, device):
    """读取完整模型对象，映射到目标设备并切换到推理模式。"""
    # 完整模型对象可能引用 __main__ 中的网络类，加载时注册对应类名。
    __main__.ResidualBlock = ResidualBlock
    __main__.TabularResNet = TabularResNet
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    return model


class ColTimeSample(Dataset):
    """每个元素保存一个交易日完整的 (X, Y) 股票截面。"""
    def __init__(self):
        self.groups = []

    def append(self, x, y):
        if len(x) != len(y):
            raise ValueError('X 和 Y 的样本数必须一致')
        self.groups.append((x, y))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


class PairwiseSoftplusLoss(nn.Module):
    """全部 i < j 股票对的 Softplus 排序损失，加预测值平方正则项。"""
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        n = pred.numel()
        # 全部 i < j 组合，包括标签相等的组合。
        upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=pred.device), diagonal=1)
        direction = torch.sign(target[None, :] - target[:, None])
        diff = pred[:, None] - pred[None, :]
        pair_losses = F.softplus(diff[upper] * direction[upper])
        ranking = pair_losses.sum() / max(n * (n - 1) // 2, 1)
        regularization = self.alpha * pred.square().mean()
        return ranking + regularization
