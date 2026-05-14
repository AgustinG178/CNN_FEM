import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchio as tio
import numpy as np
import math
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importamos la red y la función de pérdida que ya demostraron ser excelentes
from src.neural_manifold.unet_topology import UNet3D
from src.neural_manifold.dataset_pde import FocalDiceLoss

def train_dynamic_v3(
    data_dir: str,
    epochs: int = 40,
    max_lr: float = 1e-3,
    batch_size: int = 2,
    patch_size: int = 64
):
    print("="*60)
    print(" INICIANDO ENTRENAMIENTO V3 (TOTAL SEGMENTATOR + TORCHIO QUEUE)")
    print("="*60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dispositivo: {device}", flush=True)

    # 1. Configuración del Dataset Dinámico (Lazy Loading)
    subject_dirs = glob.glob(os.path.join(data_dir, "s*"))
    if not subject_dirs:
        raise ValueError(f"No se encontraron pacientes en {data_dir}")
        
    print(f"[*] Cargando {len(subject_dirs)} pacientes en la cola dinámica...", flush=True)
    
    subjects = []
    for s_dir in subject_dirs:
        ct_path = os.path.join(s_dir, "ct.nii.gz")
        mask_path = os.path.join(s_dir, "bone_mask.nii.gz")
        if os.path.exists(ct_path) and os.path.exists(mask_path):
            subject = tio.Subject(
                ct=tio.ScalarImage(ct_path),
                label=tio.LabelMap(mask_path)
            )
            subjects.append(subject)

    # Transformación personalizada para asegurar que la imagen no sea más chica que el parche
    class EnsureMinShape(tio.Transform):
        def __init__(self, min_shape):
            super().__init__()
            self.min_shape = np.array(min_shape)
            
        def apply_transform(self, subject):
            shape = np.array(subject.spatial_shape)
            if np.any(shape < self.min_shape):
                pad_size = np.maximum(0, self.min_shape - shape)
                pad_left = pad_size // 2
                pad_right = pad_size - pad_left
                # Padding tuple format for torchio: (w_left, w_right, h_left, h_right, d_left, d_right)
                padding = (pad_left[0], pad_right[0], pad_left[1], pad_right[1], pad_left[2], pad_right[2])
                
                # Para la tomografía (ct) rellenamos con -1000 (Hounsfield de aire)
                # Para la máscara (label) rellenamos con 0 (fondo)
                subject = tio.Pad(padding)(subject) # Por defecto rellena con 0 o valores mínimos
            return subject

    # Transformación para unificar matrices espaciales (Arregla error de TotalSegmentator)
    class EnforceConsistentAffine(tio.Transform):
        def apply_transform(self, subject):
            # Copia la matriz exacta del CT a la máscara para evitar que Torchio
            # colapse por diferencias en el 5to decimal (precision loss).
            # Esto fuerza la carga en RAM, pero como estamos en un hilo worker de la Queue, es perfecto.
            subject['label'] = tio.LabelMap(
                tensor=subject['label'].data, 
                affine=subject['ct'].affine
            )
            return subject

    # Aumento de datos (se aplica al volumen entero o al parche dinámicamente)
    transform = tio.Compose([
        EnforceConsistentAffine(),
        EnsureMinShape((patch_size, patch_size, patch_size)),
        tio.RandomNoise(std=0.05),
        tio.RandomFlip(axes=(0,))
    ])
    
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)

    # El muestreador extrae parches de 128x128x128.
    # label_probabilities={0: 0.1, 1: 0.9} significa que el 90% de las veces
    # el parche estará centrado en un hueso (clase 1).
    # 95% de los parches estarán centrados en vóxeles de hueso (clase 1)
    # Esto elimina los spikes de loss=1.0 causados por parches de fondo puro
    sampler = tio.data.LabelSampler(
        patch_size=patch_size, 
        label_name='label', 
        label_probabilities={0: 0.05, 1: 0.95}
    )

    # La Queue carga tomografías en RAM y les extrae parches en segundo plano
    queue = tio.Queue(
        subjects_dataset,
        max_length=200,          # Máximo de parches en RAM al mismo tiempo
        samples_per_volume=2,    # 2 parches por tomografía: balance velocidad/diversidad en CPU
        sampler=sampler,
        num_workers=6,           # Hilos de CPU preparando parches
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    # El DataLoader de PyTorch lee directo de la Queue
    train_loader = DataLoader(queue, batch_size=batch_size, num_workers=0) # num_workers=0 porque la queue ya usa hilos

    # 2. Inicialización del Modelo y Optimizador SOTA
    model = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
    criterion = FocalDiceLoss(alpha=0.8, gamma=2.0)
    
    # AdamW desacopla el weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
    
    # OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2, # 20% de las épocas serán para calentar el LR
        anneal_strategy='cos'
    )

    os.makedirs("data/03_models", exist_ok=True)
    log_path = "data/03_models/training_log_v3.csv"
    
    # 3. Bucle de Entrenamiento
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            X_batch = batch['ct'][tio.DATA].float().to(device)
            Y_batch = batch['label'][tio.DATA].float().to(device)
            
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            # Gradient clipping: evita gradientes explosivos en la fase de LR máximo
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() # El scheduler avanza en cada batch en OneCycleLR
            
            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  -> [Época {epoch+1}] Batch {batch_idx+1}/{steps_per_epoch} - Loss: {loss.item():.6f} | LR: {current_lr:.2e}", flush=True)
            
        mean_loss = epoch_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[*] Época {epoch + 1}/{epochs} finalizada - \mathcal{{L}}_{{Dice}}: {mean_loss:.6f} | LR: {current_lr:.2e}", flush=True)
        
        torch.save(model.state_dict(), f"data/03_models/unet_v3_ep{epoch+1}.pth")
        
        # Logging
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "loss", "lr"])
            writer.writerow([epoch + 1, mean_loss, current_lr])
            
        # Generar gráfico V3
        try:
            ep_list, losses, lrs = [], [], []
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep_list.append(int(row['epoch']))
                    losses.append(float(row['loss']))
                    lrs.append(float(row['lr']))

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Dice Loss', color='tab:red')
            ax1.plot(ep_list, losses, color='tab:red', marker='o', label='Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Learning Rate', color='tab:blue')
            ax2.plot(ep_list, lrs, color='tab:blue', linestyle='--', label='LR')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            plt.title('Progreso del Entrenamiento V3 (OneCycleLR)')
            fig.tight_layout()
            plt.savefig("data/03_models/loss_curve_v3.png", dpi=150)
            plt.close()
        except Exception as e:
            pass
            
    print("\n[✓] Entrenamiento V3 completado con éxito.", flush=True)

if __name__ == "__main__":
    PROCESSED_DATA_DIR = "data/05_totalsegmentator/processed"
    train_dynamic_v3(
        data_dir=PROCESSED_DATA_DIR,
        epochs=40,
        max_lr=1e-3,
        batch_size=2,
        patch_size=64
    )
