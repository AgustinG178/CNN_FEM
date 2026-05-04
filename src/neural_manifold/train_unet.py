import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.neural_manifold.unet_topology import UNet3D
from src.neural_manifold.dataset_pde import VolumetricBoneDataset, DiceLoss

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
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- AUTO-RESUME LOGIC ---
    os.makedirs("data/03_models", exist_ok=True)
    checkpoints = glob.glob("data/03_models/unet_bone_topology_ep*.pth")
    start_epoch = 0
    
    if checkpoints:
        # Encontrar la época más reciente
        latest_cp = max(checkpoints, key=lambda x: int(x.split('ep')[-1].split('.pth')[0]))
        start_epoch = int(latest_cp.split('ep')[-1].split('.pth')[0])
        print(f"-> Reanudando entrenamiento desde época {start_epoch} (cargando {latest_cp})")
        model.load_state_dict(torch.load(latest_cp, map_location=device))
    else:
        print("-> Iniciando entrenamiento desde cero.")
    
    for epoch in range(start_epoch, epochs):
        # --- PROGRAMACIÓN DEL LEARNING RATE (Cosine Annealing desde Ep 12) ---
        # Mantiene 1e-3 hasta la época 11, luego decae suavemente hasta 1e-5.
        current_lr = learning_rate
        if epoch + 1 >= 12:
            import math
            eta_min = 1e-5
            eta_max = 1e-3
            t_max = epochs - 12
            t = (epoch + 1) - 12
            current_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / t_max))
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Log del LR si hubo cambio significativo
        if epoch + 1 == 12:
            print(f"\n[!] Iniciando decaimiento Cosine Annealing. LR inicial: {current_lr:.2e}")

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

            # Monitor de progreso cada 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  -> [Época {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.6f}")
            
        mean_loss = epoch_loss / len(train_loader)
        print(f"Época {epoch + 1}/{epochs} - LR: {current_lr:.1e} - \mathcal{{L}}_{{Dice}} Promedio: {mean_loss:.6f}")
        
        # Guardado de seguridad por época
        torch.save(model.state_dict(), f"data/03_models/unet_bone_topology_ep{epoch+1}.pth")
        
        # --- NUEVO: Registro en CSV y Gráfico en Tiempo Real ---
        log_path = "data/03_models/training_log.csv"
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a') as f:
            if not file_exists:
                f.write("epoch,loss,lr\n")
            f.write(f"{epoch+1},{mean_loss},{current_lr}\n")
            
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg') # Para que no intente abrir una ventana en el clúster
            import matplotlib.pyplot as plt
            
            df = pd.read_csv(log_path)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Dice Loss', color='tab:red')
            ax1.plot(df['epoch'], df['loss'], color='tab:red', marker='o', label='Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Learning Rate', color='tab:blue')
            ax2.plot(df['epoch'], df['lr'], color='tab:blue', linestyle='--', label='LR')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            plt.title('Progreso del Entrenamiento (Real-Time)')
            fig.tight_layout()
            plt.savefig("data/03_models/loss_curve.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  [!] No se pudo generar el gráfico: {e}")
            
    os.makedirs("data/03_models", exist_ok=True)
    torch.save(model.state_dict(), "data/03_models/unet_bone_topology.pth")
    print("\n-> Entrenamiento finalizado y modelo guardado en data/03_models/unet_bone_topology.pth")

if __name__ == "__main__":
    import os
    import glob
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PATCH_DIR = os.path.join(BASE_DIR, "data", "04_training_patches")
    
    tensor_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "tensors", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "masks", "*.npy")))
    
    if not tensor_paths:
        raise ValueError(f"No se encontraron tensores de entrenamiento en {PATCH_DIR}. Ejecuta prepare_dataset.py primero.")
        
    print(f"-> Inicializando Dataset con {len(tensor_paths)} parches 3D...")
    
    # Intenta usar la versión optimizada con aumento de datos (Torchio) si está disponible, 
    # sino cae en la estándar que ya recibe parches. 
    # (Como los parches ya están extraídos, usamos la versión estándar simple)
    dataset = VolumetricBoneDataset(tensor_paths, mask_paths)
    
    # Optimización conservadora: 4 workers para evitar saturar el bus de datos
    train_loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    print("-> Iniciando Optimización del Manifold...")
    execute_optimization_manifold(train_loader, epochs=50, learning_rate=1e-3)