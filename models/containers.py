from torch import nn
from tools.typing import *


class DWConv(nn.Module):
    def __init__(self, dim=256):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.act = nn.ReLU()

        # self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.dwconv.weight)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.dwconv(x)
        x = x.permute(2, 0, 1)
        x = self.act(x)

        return x

#a mlp layer, fusion of coarse-grained features and fine-grained features
class Fusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, out_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.zeros_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        return self.fusion(x)