# -*- coding: utf-8 -*-
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

# Importamos la red, la funcion de perdida y el nuevo optimizador SAM
from src.neural_manifold.unet_topology import UNet3D
from src.neural_manifold.dataset_pde import FocalDiceLoss
from src.optimizers.sam import SAM

def train_v31_sam(
    data_dir,
    epochs=40,
    max_lr=1e-3,
    batch_size=2,
    patch_size=128  # Usamos 128 para igualar la calidad de V3.0
):
    version_id = "v3.1_SAM_OneCycle"
    output_dir = f"data/03_models/{version_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.csv")

    print("="*60)
    print(f" INICIANDO EXPERIMENTO {version_id}")
    print(f" Arquitectura: UNet3D | Opt: SAM(AdamW) | Sched: OneCycle")
    print("="*60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dispositivo: {device}", flush=True)

    # 1. Configuración del Dataset (Misma lógica que V3.0)
    subject_dirs = glob.glob(os.path.join(data_dir, "s*"))
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

    print(f"[*] Cargando {len(subjects)} pacientes...", flush=True)

    # Transformación personalizada para asegurar que la imagen no sea mas chica que el parche
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
                padding = (pad_left[0], pad_right[0], pad_left[1], pad_right[1], pad_left[2], pad_right[2])
                subject = tio.Pad(padding)(subject)
            return subject

    # Transformación para unificar matrices espaciales
    class EnforceConsistentAffine(tio.Transform):
        def apply_transform(self, subject):
            subject['label'] = tio.LabelMap(tensor=subject['label'].data, affine=subject['ct'].affine)
            return subject

    transform = tio.Compose([
        EnforceConsistentAffine(),
        EnsureMinShape((patch_size, patch_size, patch_size)),
        tio.RandomNoise(std=0.05),
        tio.RandomFlip(axes=(0,))
    ])
    
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    sampler = tio.data.LabelSampler(
        patch_size=patch_size, 
        label_name='label', 
        label_probabilities={0: 0.05, 1: 0.95}
    )

    queue = tio.Queue(
        subjects_dataset,
        max_length=100,          # Ajustado para no saturar RAM con parches de 128^3
        samples_per_volume=2,
        sampler=sampler,
        num_workers=6,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    train_loader = DataLoader(queue, batch_size=batch_size, num_workers=0)

    # 2. Inicialización de Modelo y Optimizador SAM
    model = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
    criterion = FocalDiceLoss(alpha=0.8, gamma=2.0)
    
    # SAM envuelve a AdamW
    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=max_lr, weight_decay=1e-4)
    
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer.base_optimizer, # Importante: el scheduler actúa sobre el optimizador base
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos'
    )

    # 3. Bucle de Entrenamiento con lógica SAM (2 pasos por batch)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            X = batch['ct'][tio.DATA].float().to(device)
            Y = batch['label'][tio.DATA].float().to(device)
            
            # --- PRIMER PASO SAM (Ascenso al área 'sharp') ---
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # --- SEGUNDO PASO SAM (Descenso en el punto 'sharp') ---
            Y_pred_2 = model(X)
            loss_2 = criterion(Y_pred_2, Y)
            loss_2.backward()
            optimizer.second_step(zero_grad=True)
            
            # Actualizar scheduler (se hace una vez por batch)
            scheduler.step()
            
            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  -> [Ep {epoch+1}] Batch {batch_idx+1}/{steps_per_epoch} - Loss: {loss.item():.6f} | LR: {current_lr:.2e}", flush=True)
            
        mean_loss = epoch_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[*] Época {epoch+1}/{epochs} - Loss: {mean_loss:.6f} | LR: {current_lr:.2e}", flush=True)
        
        # Guardar Checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"unet_v31_ep{epoch+1}.pth"))
        
        # Logging y Gráficos
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0: writer.writerow(["epoch", "loss", "lr"])
            writer.writerow([epoch + 1, mean_loss, current_lr])
            
        # Generar gráfico cada época
        try:
            data = np.genfromtxt(log_path, delimiter=',', names=True)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data['epoch'], data['loss'], 'tab:red', marker='o', label='Loss')
            ax1.set_ylabel('Dice Loss', color='tab:red')
            ax2 = ax1.twinx()
            ax2.plot(data['epoch'], data['lr'], 'tab:blue', linestyle='--', label='LR')
            ax2.set_ylabel('Learning Rate', color='tab:blue')
            plt.title(f'Convergencia {version_id}')
            plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
            plt.close()
        except: pass
            
    print(f"\n[✓] Experimento {version_id} finalizado.", flush=True)

if __name__ == "__main__":
    train_v31_sam(
        data_dir="data/05_totalsegmentator/processed",
        epochs=40,
        max_lr=1e-3,
        batch_size=2,
        patch_size=128
    )
