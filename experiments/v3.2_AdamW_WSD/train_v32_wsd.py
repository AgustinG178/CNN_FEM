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

# Importamos la red, la pérdida y el nuevo scheduler WSD
from src.neural_manifold.unet_topology import UNet3D
from src.neural_manifold.dataset_pde import FocalDiceLoss
from src.schedulers.wsd import WSDScheduler

def train_v32_wsd(
    data_dir: str,
    epochs: int = 40,
    max_lr: float = 1e-3,
    batch_size: int = 2,
    patch_size: int = 128
):
    version_id = "v3.2_AdamW_WSD"
    output_dir = f"data/03_models/{version_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.csv")

    print("="*60)
    print(f" INICIANDO EXPERIMENTO {version_id}")
    print(f" Arquitectura: UNet3D | Opt: AdamW | Sched: WSD (10-70-20)")
    print("="*60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dispositivo: {device}", flush=True)

    # 1. Configuración del Dataset (Mismo que V3.0 y V3.1)
    subject_dirs = glob.glob(os.path.join(data_dir, "s*"))
    subjects = []
    for s_dir in subject_dirs:
        ct_path = os.path.join(s_dir, "ct.nii.gz")
        mask_path = os.path.join(s_dir, "bone_mask.nii.gz")
        if os.path.exists(ct_path) and os.path.exists(mask_path):
            subject = tio.Subject(ct=tio.ScalarImage(ct_path), label=tio.LabelMap(mask_path))
            subjects.append(subject)

    transform = tio.Compose([
        tio.RandomNoise(std=0.05),
        tio.RandomFlip(axes=(0,))
    ])
    
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    sampler = tio.data.LabelSampler(patch_size=patch_size, label_name='label', label_probabilities={0: 0.05, 1: 0.95})

    queue = tio.Queue(
        subjects_dataset,
        max_length=100,
        samples_per_volume=2,
        sampler=sampler,
        num_workers=6,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    train_loader = DataLoader(queue, batch_size=batch_size, num_workers=0)

    # 2. Inicialización de Modelo y Scheduler WSD
    model = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
    criterion = FocalDiceLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
    
    total_steps = epochs * len(train_loader)
    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=int(total_steps * 0.1),  # 10% subida
        stable_steps=int(total_steps * 0.8),  # 70% estable (hasta el 80% total)
        total_steps=total_steps,
        min_lr=1e-6
    )

    # 3. Bucle de Entrenamiento
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            X = batch['ct'][tio.DATA].float().to(device)
            Y = batch['label'][tio.DATA].float().to(device)
            
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  -> [Ep {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.6f} | LR: {current_lr:.2e}", flush=True)
            
        mean_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[*] Época {epoch+1}/{epochs} - Loss: {mean_loss:.6f} | LR: {current_lr:.2e}", flush=True)
        
        torch.save(model.state_dict(), os.path.join(output_dir, f"unet_v32_ep{epoch+1}.pth"))
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0: writer.writerow(["epoch", "loss", "lr"])
            writer.writerow([epoch + 1, mean_loss, current_lr])
            
        # Gráfico
        try:
            data = np.genfromtxt(log_path, delimiter=',', names=True)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data['epoch'], data['loss'], 'tab:red', marker='o', label='Loss')
            ax2 = ax1.twinx()
            ax2.plot(data['epoch'], data['lr'], 'tab:blue', linestyle='--', label='LR')
            plt.title(f'Convergencia {version_id}')
            plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
            plt.close()
        except: pass
            
    print(f"\n[✓] Experimento {version_id} finalizado.", flush=True)

if __name__ == "__main__":
    train_v32_wsd(
        data_dir="data/05_totalsegmentator/processed",
        epochs=40,
        max_lr=1e-3,
        batch_size=2,
        patch_size=128
    )
