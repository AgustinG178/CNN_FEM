"""
train_unet_v4.py
----------------
Motor de Entrenamiento V4 — BoneFlow AI
GPU: 2× NVIDIA T4 (Google Cloud Platform)
Modo: DDP (Distributed Data Parallel) — Multi-GPU

Diferencias clave respecto a V3:
  ✅ DDP multi-GPU (2× T4 = efectivo 32GB VRAM)
  ✅ Automatic Mixed Precision fp16 (AMP)
  ✅ Patch size 256³ (máxima calidad de contexto anatómico)
  ✅ Validation Loop real: Dice Score honesto en cada época
  ✅ Early Stopping + Best Model Checkpoint
  ✅ Lee dataset_split.json (reproducibilidad científica)

Cómo lanzar (en la VM de GCP):
  torchrun --nproc_per_node=2 src/neural_manifold/train_unet_v4.py
"""

import os
import json
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import torchio as tio
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from src.neural_manifold.unet_topology import UNet3D
from src.neural_manifold.dataset_pde import FocalDiceLoss

# =====================================================================
# HIPERPARAacuteMETROS V4 — Lambda Labs V100 (88 vCPUs / 448GB RAM)
# =====================================================================
PATCH_SIZE          = 256    # 256³ fp16: ~6GB por batch=4 (entra en 16GB V100)
BATCH_SIZE          = 4      # Sin DDP, V100 puede con bs=4 en fp16
EPOCHS             = 40
MAX_LR              = 3e-3
WEIGHT_DECAY        = 1e-4
SAMPLES_PER_VOL     = 8
QUEUE_MAX_LENGTH    = 500    # 448GB RAM: podemos tener más parches en memoria
NUM_WORKERS         = 32     # 88 vCPUs: máximo paralelismo de I/O sin costo
EARLY_STOP_PATIENCE = 8

SPLIT_PATH      = "data/05_totalsegmentator/dataset_split.json"
MODEL_DIR       = "data/03_models"
BEST_MODEL_PATH = "data/03_models/unet_v4_best.pth"
LOG_PATH        = "data/03_models/training_log_v4.csv"
CURVE_PATH      = "data/03_models/loss_curve_v4.png"
# =====================================================================


def dice_score_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin     = (pred.sigmoid() > threshold).float()
    intersection = (pred_bin * target).sum()
    denom        = pred_bin.sum() + target.sum()
    if denom == 0:
        return 1.0
    return (2. * intersection / denom).item()


class EnforceConsistentAffine(tio.Transform):
    def apply_transform(self, subject):
        subject['label'] = tio.LabelMap(
            tensor=subject['label'].data,
            affine=subject['ct'].affine
        )
        return subject


class EnsureMinShape(tio.Transform):
    def __init__(self, min_shape):
        super().__init__()
        self.min_shape = np.array(min_shape)

    def apply_transform(self, subject):
        shape = np.array(subject.spatial_shape)
        if np.any(shape < self.min_shape):
            pad_size  = np.maximum(0, self.min_shape - shape)
            pad_left  = pad_size // 2
            pad_right = pad_size - pad_left
            padding   = tuple(np.array([pad_left, pad_right]).T.flatten())
            subject   = tio.Pad(padding)(subject)
        return subject


def build_subjects(patient_list: list) -> list:
    subjects = []
    for p in patient_list:
        ct_path   = p["ct_path"]
        mask_path = p["mask_path"]
        if os.path.exists(ct_path) and os.path.exists(mask_path):
            subjects.append(tio.Subject(
                ct=tio.ScalarImage(ct_path),
                label=tio.LabelMap(mask_path)
            ))
    return subjects


def build_queue(subjects, augment=True):
    transforms = [
        EnforceConsistentAffine(),
        EnsureMinShape((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)),
    ]
    if augment:
        transforms += [
            tio.RandomNoise(std=0.05),
            tio.RandomFlip(axes=(0,)),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        ]

    dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms))
    sampler = tio.data.LabelSampler(
        patch_size=PATCH_SIZE,
        label_name='label',
        label_probabilities={0: 0.05, 1: 0.95}
    )
    return tio.Queue(
        dataset,
        max_length=QUEUE_MAX_LENGTH,
        samples_per_volume=SAMPLES_PER_VOL,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        shuffle_subjects=True,
        shuffle_patches=True,
    )


def plot_curves(log_path, out_path):
    epochs, losses, val_dices, lrs = [], [], [], []
    try:
        with open(log_path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row['epoch']))
                losses.append(float(row['train_loss']))
                val_dices.append(float(row['val_dice']))
                lrs.append(float(row['lr']))
    except Exception:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Train Loss / Val Dice')
    ax1.plot(epochs, losses,    'r-o', label='Train Loss (↓)', markersize=4)
    ax1.plot(epochs, val_dices, 'g-s', label='Val Dice Score (↑)', markersize=4)
    ax1.axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Objetivo 85%')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:blue')
    ax2.plot(epochs, lrs, 'b--', alpha=0.5, label='LR')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'BoneFlow V4 — 2× T4 DDP | 256³ fp16 | Mejor Val Dice: {max(val_dices):.4f}')
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_worker(rank: int, world_size: int):
    """Función principal de entrenamiento para cada proceso GPU."""
    setup_ddp(rank, world_size)

    if is_main_process(rank):
        print("=" * 60)
        print(f" BONEFLOW AI — ENTRENAMIENTO V4 (DDP)")
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" GPUs: {world_size} × {torch.cuda.get_device_name(rank)}")
        print(f" Patch: {PATCH_SIZE}³ | bs/GPU: {BATCH_SIZE_PER_GPU} | Efectivo: {BATCH_SIZE_PER_GPU*world_size}")
        print("=" * 60, flush=True)

    device = torch.device(f'cuda:{rank}')

    # 1. Cargar el split
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    train_subjects = build_subjects(split["train"])
    val_subjects   = build_subjects(split["validation"])

    if is_main_process(rank):
        print(f"[*] Train: {len(train_subjects)} | Val: {len(val_subjects)}", flush=True)

    # 2. Queues (cada GPU tiene su propia Queue en DDP)
    train_queue = build_queue(train_subjects, augment=True)
    val_queue   = build_queue(val_subjects,   augment=False)

    train_loader = DataLoader(train_queue, batch_size=BATCH_SIZE_PER_GPU, num_workers=0)
    val_loader   = DataLoader(val_queue,   batch_size=1,                  num_workers=0)

    steps_per_epoch = len(train_loader)

    # 3. Modelo + DDP wrapper
    model     = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
    model     = DDP(model, device_ids=[rank])
    criterion = FocalDiceLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.2, anneal_strategy='cos'
    )
    scaler = GradScaler()

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_dice    = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            X = batch['ct'][tio.DATA].float().to(device)
            Y = batch['label'][tio.DATA].float().to(device)

            optimizer.zero_grad()
            with autocast():
                Y_pred = model(X)
                loss   = criterion(Y_pred, Y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if is_main_process(rank) and (batch_idx + 1) % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  -> [Época {epoch+1}] Batch {batch_idx+1}/{steps_per_epoch} "
                      f"- Loss: {loss.item():.6f} | LR: {lr:.2e}", flush=True)

        mean_loss = epoch_loss / steps_per_epoch

        # --- VALIDATION (solo en GPU 0 para no duplicar) ---
        if is_main_process(rank):
            model.eval()
            val_dice_sum = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    X_val = val_batch['ct'][tio.DATA].float().to(device)
                    Y_val = val_batch['label'][tio.DATA].float().to(device)
                    with autocast():
                        Y_pred_val = model(X_val)
                    val_dice_sum += dice_score_metric(Y_pred_val.cpu(), Y_val.cpu())

            mean_val_dice = val_dice_sum / len(val_loader)
            lr_now        = optimizer.param_groups[0]['lr']

            print(f"[✓] Época {epoch+1}/{EPOCHS} — "
                  f"Train Loss: {mean_loss:.6f} | "
                  f"Val Dice: {mean_val_dice:.4f} ({mean_val_dice*100:.1f}%) | "
                  f"LR: {lr_now:.2e}", flush=True)

            # Checkpoint
            if mean_val_dice > best_val_dice:
                best_val_dice    = mean_val_dice
                patience_counter = 0
                torch.save(model.module.state_dict(), BEST_MODEL_PATH)
                print(f"  ★ Nuevo mejor modelo! Val Dice: {best_val_dice*100:.1f}%", flush=True)
            else:
                patience_counter += 1

            torch.save(model.module.state_dict(), f"{MODEL_DIR}/unet_v4_ep{epoch+1}.pth")

            # Logging
            file_exists = os.path.isfile(LOG_PATH)
            with open(LOG_PATH, 'a', newline='') as f_log:
                writer = csv.writer(f_log)
                if not file_exists:
                    writer.writerow(["epoch", "train_loss", "val_dice", "lr"])
                writer.writerow([epoch+1, mean_loss, mean_val_dice, lr_now])

            plot_curves(LOG_PATH, CURVE_PATH)

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[!] Early Stopping. Sin mejora en {EARLY_STOP_PATIENCE} épocas.", flush=True)
                break

        # Sincronizar todos los procesos antes de la siguiente época
        dist.barrier()

    if is_main_process(rank):
        print(f"\n[✓] V4 completado. Mejor Val Dice: {best_val_dice*100:.1f}%", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No se detectaron GPUs. Verificá la instalación de CUDA.")

    print(f"[*] Iniciando DDP con {world_size} GPU(s)...")
    # torchrun maneja el spawn automáticamente
    # Lanzar con: torchrun --nproc_per_node=2 src/neural_manifold/train_unet_v4.py
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_worker(rank, world_size)
