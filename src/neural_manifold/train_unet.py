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
    
    for epoch in range(epochs):
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
            
        mean_loss = epoch_loss / len(train_loader)
        print(f"Época {epoch + 1}/{epochs} - \mathcal{{L}}_{{Dice}} Promedio: {mean_loss:.6f}")
        
        # Auto-guardado de seguridad por época (por si lo interrumpes antes de tiempo)
        os.makedirs("data/03_models", exist_ok=True)
        torch.save(model.state_dict(), f"data/03_models/unet_bone_topology_ep{epoch+1}.pth")
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
    
    # Aumentamos los workers a 8 para sacarle provecho a los 24 cores del clúster
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)
    
    print("-> Iniciando Optimización del Manifold...")
    execute_optimization_manifold(train_loader, epochs=50, learning_rate=1e-3)