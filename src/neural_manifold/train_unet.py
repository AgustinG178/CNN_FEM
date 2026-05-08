import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.neural_manifold.unet_topology import UNet3D
from src.neural_manifold.dataset_pde import VolumetricBoneDataset, FocalDiceLoss
import os
import glob
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

def execute_optimization_manifold(
    train_loader: DataLoader, 
    epochs: int = 50, 
    learning_rate: float = 1e-3, 
    device_str: str = 'cuda'
) -> None:
    r"""
    Evalúa la convergencia del modelo \mathcal{M}_\theta minimizando 
    la divergencia topológica \mathcal{L}_{Dice} sobre el espacio \mathbb{R}^3.
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
    criterion = FocalDiceLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- AUTO-RESUME LOGIC ---
    os.makedirs("data/03_models", exist_ok=True)
    checkpoints = glob.glob("data/03_models/unet_bone_topology_ep*.pth")
    start_epoch = 0
    
    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(x.split('ep')[-1].split('.pth')[0]))
        start_epoch = int(latest_cp.split('ep')[-1].split('.pth')[0])
        print(f"-> Reanudando entrenamiento desde época {start_epoch} (cargando {latest_cp})", flush=True)
        model.load_state_dict(torch.load(latest_cp, map_location=device))
    else:
        print("-> Iniciando entrenamiento desde cero.", flush=True)
    
    for epoch in range(start_epoch, epochs):
        # --- PROGRAMACIÓN DEL LEARNING RATE (Cosine Annealing desde Ep 12) ---
        current_lr = learning_rate
        if epoch + 1 >= 12:
            eta_min = 1e-5
            eta_max = 1e-3
            t_max = epochs - 12
            t = (epoch + 1) - 12
            current_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / t_max))
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        if epoch + 1 == 12:
            print(f"\n[!] Iniciando decaimiento Cosine Annealing. LR inicial: {current_lr:.2e}", flush=True)

        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  -> [Época {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.6f}", flush=True)
            
        mean_loss = epoch_loss / len(train_loader)
        print(f"Época {epoch + 1}/{epochs} - LR: {current_lr:.1e} - \mathcal{{L}}_{{Dice}} Promedio: {mean_loss:.6f}", flush=True)
        
        torch.save(model.state_dict(), f"data/03_models/unet_bone_topology_ep{epoch+1}.pth")
        
        # --- Registro en CSV y Gráfico en Tiempo Real (Versión sin Pandas) ---
        log_path = "data/03_models/training_log.csv"
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "loss", "lr"])
            writer.writerow([epoch + 1, mean_loss, current_lr])
            
        try:
            epochs_list, losses, lrs = [], [], []
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs_list.append(int(row['epoch']))
                    losses.append(float(row['loss']))
                    lrs.append(float(row['lr']))

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Dice Loss', color='tab:red')
            ax1.plot(epochs_list, losses, color='tab:red', marker='o', label='Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Learning Rate', color='tab:blue')
            ax2.plot(epochs_list, lrs, color='tab:blue', linestyle='--', label='LR')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            plt.title('Progreso del Entrenamiento (Real-Time)')
            fig.tight_layout()
            plt.savefig("data/03_models/loss_curve.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  [!] No se pudo generar el gráfico: {e}", flush=True)
            
    torch.save(model.state_dict(), "data/03_models/unet_bone_topology.pth")
    print("\n-> Entrenamiento finalizado y modelo guardado.", flush=True)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PATCH_DIR = os.path.join(BASE_DIR, "data", "04_training_patches")
    
    tensor_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "tensors", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "masks", "*.npy")))
    
    if not tensor_paths:
        raise ValueError(f"No se encontraron tensores en {PATCH_DIR}.")
        
    dataset = VolumetricBoneDataset(tensor_paths, mask_paths)
    train_loader = DataLoader(
        dataset, 
        batch_size=2,  # Reducido por el enorme tamaño del parche 128^3 y base_features=32
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    execute_optimization_manifold(train_loader, epochs=50, learning_rate=1e-3)