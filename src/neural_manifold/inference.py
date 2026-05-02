import os
import torch
import torchio as tio
import numpy as np
from src.tensor_pde.io_module import assemble_tensor_and_hu
from src.neural_manifold.unet_topology import UNet3D

def predict_volume_from_dicom(
    dicom_dir: str,
    model_path: str,
    patch_size: tuple = (64, 64, 64),
    patch_overlap: tuple = (16, 16, 16),
    device_str: str = 'cuda',
    return_probabilities: bool = False
) -> np.ndarray:
    r"""
    Orquesta la inferencia sobre volúmenes masivos mediante reconstrucción por parches deslizantes,
    preservando la invariancia topológica y maximizando la resolución local.
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"-> Cargando modelo topológico desde: {model_path} en {device}")
    
    # 1. Instanciar la red y cargar pesos
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[!] ADVERTENCIA: No se encontró {model_path}. Se usarán pesos aleatorios para prueba.")
        
    # [!] HACK DE INGENIERÍA: En lugar de model.eval(), forzamos model.train()
    # Esto obliga a las capas BatchNorm a recalcular la media y varianza sobre el
    # parche actual, en lugar de usar la "running_mean" sesgada del entrenamiento.
    model.train()

    # 2. Ensamblar Tensor 3D a partir del DICOM y mapeo HU
    print("-> Ensamblando tensor espacial desde DICOM...")
    X_np = assemble_tensor_and_hu(dicom_dir)
    
    # Mapeo a dominio [0, 1] idéntico a VolumetricBoneDataset
    X_np = np.clip(X_np, a_min=-1000.0, a_max=3000.0)
    X_np = (X_np + 1000.0) / 4000.0
    
    # 3. Preparación de TorchIO para inferencia particionada (Sliding Window)
    subject = tio.Subject(
        volume=tio.ScalarImage(tensor=torch.from_numpy(X_np).float().unsqueeze(0))
    )
    
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
    
    # 4. Inferencia Local iterativa
    print(f"-> Computando inferencia topológica mediante parches deslizantes {patch_size}...")
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['volume'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            
            logits = model(inputs)
            
            # Agregamos las predicciones locales al espacio global
            aggregator.add_batch(logits.cpu(), locations)
            
    # 5. Reconstrucción global
    predicted_volume = aggregator.get_output_tensor().squeeze().numpy()
    print("-> Reconstrucción del co-dominio espacial finalizada.")
    
    if return_probabilities:
        return predicted_volume
        
    # Binarización usando umbral
    binary_mask = (predicted_volume > 0.5)
    
    return binary_mask
