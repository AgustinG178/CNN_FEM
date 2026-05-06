import torch
import numpy as np
import matplotlib.pyplot as plt
from src.neural_manifold.unet_topology import UNet3D
import glob
import os

# CONFIGURACIÓN
EPOCA = 26
MODEL_PATH = f"data/03_models/unet_bone_topology_ep{EPOCA}.pth"
PATCH_DIR = "data/04_training_patches"

def debug_patch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar modelo
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. Agarrar un parche de entrenamiento al azar
    tensors = sorted(glob.glob(os.path.join(PATCH_DIR, "tensors", "*.npy")))
    masks = sorted(glob.glob(os.path.join(PATCH_DIR, "masks", "*.npy")))
    
    idx = len(tensors) // 2 # Agarramos uno del medio
    X = np.load(tensors[idx])
    Y_gt = np.load(masks[idx])
    
    # 3. Inferencia sobre el parche
    X_torch = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        Y_pred = model(X_torch).cpu().squeeze().numpy()
    
    # 4. Visualización
    slice_idx = 32
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(X[slice_idx], cmap='gray')
    axes[0].set_title("Tomografía (Input)")
    
    axes[1].imshow(Y_gt[slice_idx], cmap='jet')
    axes[1].set_title("Máscara Real (Ground Truth)")
    
    axes[2].imshow(Y_pred[slice_idx], cmap='jet')
    axes[2].set_title(f"Predicción IA (Época {EPOCA})")
    
    plt.suptitle(f"Sanity Check - Parche {os.path.basename(tensors[idx])}")
    plt.savefig("debug_prediction.png")
    print(f"[✓] Imagen guardada en debug_prediction.png. ¡Mirala para ver si coinciden!")

if __name__ == "__main__":
    debug_patch()
