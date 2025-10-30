import torch.nn as nn

class SmallCifarNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
            )
        self.features = nn.Sequential(
            block(3, 64), block(64, 64), nn.MaxPool2d(2),
            block(64,128), block(128,128), nn.MaxPool2d(2),
            block(128,256), block(256,256), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
