import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import DCNNConfig


class DCNN(nn.Module):
    def __init__(self, config: DCNNConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.conv2_list = nn.ModuleList(
            [nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3_list = nn.ModuleList(
            [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_list = nn.ModuleList(
            [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_dropout = nn.Dropout(config.conv_dropout)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.fc1 = nn.Linear(128 * (config.image_size // 16) ** 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        c1_out = self.pool(F.relu(self.conv1(x)))  # 64

        c2_out = c1_out.clone()
        for conv2 in self.conv2_list:
            c2_out = conv2(c2_out)
        c2_out = self.conv_dropout(self.pool(F.relu(c2_out + c1_out)))  # 32

        c3 = self.conv3(c2_out)
        c3_out = c3.clone()
        for conv3 in self.conv3_list:
            c3_out = conv3(c3_out)
        c3_out = self.conv_dropout(self.pool(F.relu(c3_out + c3)))  # 16

        c4 = self.conv4(c3_out)
        c4_out = c4.clone()
        for conv4 in self.conv4_list:
            c4_out = conv4(c4_out)
        c4_out = self.conv_dropout(self.pool(F.relu(c4_out + c4)))  # 8

        r = torch.flatten(c4_out, 1)  # flatten all dimensions except batch
        r = self.fc_dropout(F.relu(self.fc1(r)))
        r = self.fc_dropout(F.relu(self.fc2(r)))
        r = self.fc3(r)
        return r
