import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset

class HybridLoss(nn.Module):
    r"""
    Operador de pérdida híbrido: \alpha Dice + (1-\alpha) CrossEntropy.
    Diseñado para segmentación multidominio en el espacio \mathbb{R}^3.
    """
    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true debe ser un tensor de índices de clase [B, D, H, W]
        # y_pred debe ser [B, K, D, H, W]
        
        ce_loss = self.cross_entropy(y_pred, y_true.long())
        
        # Convertimos y_true a One-Hot para el producto interno del Dice
        K = y_pred.shape[1]
        y_true_one_hot = F.one_hot(y_true.long(), num_classes=K).permute(0, 4, 1, 2, 3).float()
        
        y_pred_soft = F.softmax(y_pred, dim=1)
        
        dims = (2, 3, 4)
        intersection = torch.sum(y_pred_soft * y_true_one_hot, dim=dims)
        cardinality = torch.sum(y_pred_soft + y_true_one_hot, dim=dims)
        
        dice_score = (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
        dice_loss = 1. - dice_score.mean()
        
        return self.alpha * dice_loss + (1.0 - self.alpha) * ce_loss

class DiceLoss(nn.Module):
    r"""
    Computa el error topológico mediante el coeficiente de Dice diferenciable.
    \mathcal{L}_{Dice} = 1 - \frac{2|X \cap Y| + \epsilon}{|X| + |Y| + \epsilon}
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Aplanar los tensores para calcular la intersección
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1.0 - dice

class VolumetricBoneDataset(Dataset):
    r"""
    Dataset optimizado con aumento de datos elástico para 
    la invariancia morfológica de la pelvis.
    """
    def __init__(self, tensor_paths: list, mask_paths: list, augment: bool = True):
        self.tensor_paths = tensor_paths
        self.mask_paths = mask_paths
        
        # Definición del operador de aumento T
        self.transform = tio.Compose([
            tio.RandomElasticDeformation(num_control_points=7, max_displacement=10),
            tio.RandomBiasField(), # Simula heterogeneidad del campo magnético/rayos X
            tio.RandomNoise(std=0.05),
            tio.RandomFlip(axes=(0,)) # Simetría especular (Izquierda/Derecha)
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.tensor_paths)

    def __getitem__(self, idx: int):
        X_np = np.load(self.tensor_paths[idx])
        Y_np = np.load(self.mask_paths[idx])
        
        # Empaquetado en un objeto Subject de Torchio
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(X_np).float().unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.from_numpy(Y_np).float().unsqueeze(0))
        )
        
        if self.transform:
            subject = self.transform(subject)
            
        return subject.image.data, subject.label.data