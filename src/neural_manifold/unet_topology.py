import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    r"""
    Bloque Residual 3D con Instance Normalization.
    Permite el flujo directo del gradiente a través de 'shortcuts', 
    preservando detalles topológicos de alta frecuencia (cortical ósea fina).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        # Shortcut para alinear canales si es necesario
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual  # Fusión de identidad
        return self.relu(out)

class AttentionGate3D(nn.Module):
    r"""
    Mecanismo de Atención Espacial.
    Utiliza las características de baja resolución (gating signal 'g') para 
    enfocar las características de alta resolución ('x') en las regiones anatómicas relevantes,
    suprimiendo activaciones de fondo inútiles.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi) # Mapa de atención [0, 1]
        return x * psi # Filtrado espacial

class UNet3D(nn.Module):
    r"""
    Attention-ResUNet3D (SOTA Medical Image Segmentation).
    Integra bloques residuales para extracción profunda de características y 
    compuertas de atención para recuperar bordes topológicos finos con precisión.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        
        # --- ENCODER (Contracción Residual) ---
        self.down1 = ResidualBlock3D(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.down2 = ResidualBlock3D(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.down3 = ResidualBlock3D(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck (Espacio latente de mayor compresión)
        self.bottleneck = ResidualBlock3D(base_features * 4, base_features * 8)
        
        # --- DECODER (Expansión con Atención) ---
        self.up1 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(F_g=base_features * 4, F_l=base_features * 4, F_int=base_features * 2)
        self.conv_up1 = ResidualBlock3D(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(F_g=base_features * 2, F_l=base_features * 2, F_int=base_features)
        self.conv_up2 = ResidualBlock3D(base_features * 4, base_features * 2)
        
        self.up3 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(F_g=base_features, F_l=base_features, F_int=base_features // 2)
        self.conv_up3 = ResidualBlock3D(base_features * 2, base_features)
        
        # Salida
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        
        # Bottleneck
        bn = self.bottleneck(p3)
        
        # Decoder 1
        u1 = self.up1(bn)
        x3_att = self.att1(g=u1, x=x3) # Atención aplicada al skip connection
        u1_cat = torch.cat([u1, x3_att], dim=1)
        u1_out = self.conv_up1(u1_cat)
        
        # Decoder 2
        u2 = self.up2(u1_out)
        x2_att = self.att2(g=u2, x=x2)
        u2_cat = torch.cat([u2, x2_att], dim=1)
        u2_out = self.conv_up2(u2_cat)
        
        # Decoder 3
        u3 = self.up3(u2_out)
        x1_att = self.att3(g=u3, x=x1)
        u3_cat = torch.cat([u3, x1_att], dim=1)
        u3_out = self.conv_up3(u3_cat)
        
        out = self.final_conv(u3_out)
        return self.sigmoid(out)