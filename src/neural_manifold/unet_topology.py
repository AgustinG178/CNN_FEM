import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    r"""
    Computa la composición de dos operadores integrales de convolución espacial 
    con núcleos \mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times 3 \times 3 \times 3}, 
    seguidos de normalización por lotes y activación no lineal estricta (ReLU).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet3D(nn.Module):
    r"""
    Instancia la arquitectura topológica \mathcal{M}_\theta para el mapeo no lineal 
    \hat{\mathbf{Y}} = \mathcal{M}_\theta(\mathbf{X}).
    Evalúa tensores espaciales \mathbf{X} \in \mathbb{R}^{B \times 1 \times D \times H \times W} 
    y proyecta un campo escalar de probabilidades \hat{\mathbf{Y}} \in [0, 1].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        
        # Ruta de Contracción (Extracción de invariantes topológicos locales)
        self.down1 = DoubleConv3D(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.down2 = DoubleConv3D(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.down3 = DoubleConv3D(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv3D(base_features * 4, base_features * 8)
        
        self.up1 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(base_features * 4, base_features * 2)
        
        self.up3 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(base_features * 2, base_features)
        
        # Proyección al codominio [0, 1] mediante función Sigmoid para clasificación binaria
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        
        bn = self.bottleneck(p3)
        
        u1 = self.up1(bn)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        
        out = self.final_conv(u3)
        return self.sigmoid(out)