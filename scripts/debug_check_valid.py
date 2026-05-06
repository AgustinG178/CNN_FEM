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

def debug_valid_patch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    tensors = sorted(glob.glob(os.path.join(PATCH_DIR, "tensors", "*.npy")))
    masks = sorted(glob.glob(os.path.join(PATCH_DIR, "masks", "*.npy")))
    
    # BUSCAMOS UN PARCHE QUE TENGA HUESO DE VERDAD
    found_idx = -1
    for i in range(len(masks)):
        m = np.load(masks[i])
        if np.sum(m) > 5000: # Que tenga una cantidad decente de vóxeles de hueso
            found_idx = i
            break
            
    if found_idx == -1:
        print("[!] No se encontró ningún parche con hueso en los primeros archivos.")
        return

    X = np.load(tensors[found_idx])
    Y_gt = np.load(masks[found_idx])
    
    X_torch = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        Y_pred = model(X_torch).cpu().squeeze().numpy()
    
    slice_idx = 32
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(X[slice_idx], cmap='gray')
    axes[0].set_title("Input (CT)")
    axes[1].imshow(Y_gt[slice_idx], cmap='jet')
    axes[1].set_title("Target (Hueso Real)")
    axes[2].imshow(Y_pred[slice_idx], cmap='jet')
    axes[2].set_title(f"Predicción (Época {EPOCA})")
    
    plt.suptitle(f"Validación sobre Parche con Hueso: {os.path.basename(tensors[found_idx])}")
    plt.savefig("debug_valid_patch.png")
    print(f"[✓] Verificando parche {found_idx}. Imagen guardada en debug_valid_patch.png")

if __name__ == "__main__":
    debug_valid_patch()
