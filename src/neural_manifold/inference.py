import os
import torch
import torchio as tio
import numpy as np
from src.tensor_pde.io_module import assemble_tensor_and_hu
from src.neural_manifold.unet_topology import UNet3D

def predict_volume_from_dicom(
    dicom_dir: str,
    model_path: str,
    patch_size: tuple = (128, 128, 128),
    patch_overlap: tuple = (32, 32, 32),
    device_str: str = 'cuda',
    return_probabilities: bool = False
) -> np.ndarray:
    r"""
    Orquesta la inferencia sobre volúmenes masivos mediante reconstrucción por parches deslizantes,
    preservando la invariancia topológica y maximizando la resolución local.
    
    SOLUCIÓN AL PROBLEMA DE BatchNorm:
    En vez de usar model.train() (que produce estadísticas inconsistentes entre parches)
    o model.eval() (que usa running_stats sesgados del entrenamiento), usamos
    RECALIBRACIÓN: primero pasamos todos los parches del volumen actual por el modelo
    en modo train para actualizar las running_stats, y luego hacemos la inferencia
    real en modo eval con las estadísticas recalibradas.
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"-> Cargando modelo topológico desde: {model_path} en {device}")
    
    # 1. Instanciar la red y cargar pesos
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[!] ADVERTENCIA: No se encontró {model_path}. Se usarán pesos aleatorios.")
        
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
    
    # =========================================================================
    # FASE 1: RECALIBRACIÓN DE BatchNorm
    # =========================================================================
    # Pasamos TODOS los parches del volumen actual por el modelo en modo train
    # para que las running_mean/running_var se actualicen con las estadísticas
    # de ESTE paciente específico. Esto resuelve el sesgo del entrenamiento.
    # =========================================================================
    print("-> Recalibrando BatchNorm sobre el volumen actual...")
    
    # Resetear running stats de todas las capas BatchNorm
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            module.running_mean.zero_()
            module.running_var.fill_(1.0)
            module.num_batches_tracked.zero_()
            # Usar momentum=0.1 para que las stats se estabilicen rápido
            module.momentum = 0.1
    
    model.train()
    calibration_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
    with torch.no_grad():
        for patches_batch in calibration_loader:
            inputs = patches_batch['volume'][tio.DATA].to(device)
            _ = model(inputs)  # Solo forward pass para actualizar running_stats
    
    print("   BatchNorm recalibrado con las estadísticas de este volumen.")
    
    # =========================================================================
    # FASE 2: INFERENCIA REAL EN MODO EVAL
    # =========================================================================
    # Ahora que las running_stats reflejan este paciente, usamos model.eval()
    # para que TODOS los parches se normalicen con las MISMAS estadísticas.
    # Esto elimina la inconsistencia entre parches.
    # =========================================================================
    model.eval()
    
    # Necesitamos un nuevo GridSampler y Aggregator (el anterior fue consumido)
    grid_sampler2 = tio.inference.GridSampler(
        subject,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(grid_sampler2, batch_size=4)
    aggregator = tio.inference.GridAggregator(grid_sampler2, overlap_mode='average')
    
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
    
    # --- FILTRO FÍSICO CRÍTICO ---
    # Usamos el tensor original X_np (que escalamos de 0 a 1) para recuperar 
    # la máscara de lo que NO es hueso físicamente (< 200 HU).
    # Re-calculamos el umbral normalizado: (200 + 1000) / 4000 = 0.3
    phys_bone_mask = (X_np > 0.3).squeeze()
    predicted_volume[phys_bone_mask == 0] = 0.0
    # -----------------------------
    
    print("-> Reconstrucción del co-dominio espacial finalizada.")
    
    if return_probabilities:
        return predicted_volume
        
    # Binarización usando umbral
    binary_mask = (predicted_volume > 0.5)
    
    return binary_mask
