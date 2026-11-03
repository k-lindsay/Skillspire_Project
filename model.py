import torch
import torch.nn as nn

class boat_detection_model(nn.Module):
    def __init__(self, input: int = 3, hiddenwork: int = 16, output: int = 5):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=hiddenwork, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hiddenwork),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.1)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenwork, out_channels=hiddenwork * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hiddenwork * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.15)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenwork * 2, out_channels=hiddenwork * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hiddenwork * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenwork * 4, out_channels=hiddenwork * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hiddenwork * 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenwork * 8 * 14 * 14, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=output)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.classifier(x)
        return x
