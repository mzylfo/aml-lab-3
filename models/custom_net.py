import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # --- Blocco 1: da 3 canali a 64 ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)

        # --- Blocco 2: da 64 canali a 128 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # --- Blocco 3: da 128 canali a 256 ---
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Regola VGG: dimezza le dimensioni spaziali (H e W)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Approccio moderno (ResNet): Global Average Pooling (porta a 1x1 pixel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classificatore finale su 200 classi (TinyImageNet)
        self.fc1 = nn.Linear(256, 200)

    def forward(self, x):
        # B x 3 x 224 x 224

        # Blocco 1: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.bn1(x).relu()
        x = self.pool(x)       # Dimezza la risoluzione

        # Blocco 2: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.bn2(x).relu()
        x = self.pool(x)       # Dimezza la risoluzione

        # Blocco 3: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.conv3(x)
        x = self.bn3(x).relu()
        x = self.pool(x)       # Dimezza la risoluzione

        # Global Average Pooling: riduce ogni canale a 1 singolo valore (B x 256 x 1 x 1)
        x = self.gap(x)

        # Appiattimento del tensore a dimensione [B, 256]
        x = torch.flatten(x, 1)

        # Layer lineare finale: assegna i punteggi alle 200 classi (B x 200)
        x = self.fc1(x)

        return x