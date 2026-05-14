"""
swa_ensemble.py  —  V3.1: Stochastic Weight Averaging
-------------------------------------------------------
Implementa la idea de Agustín: en vez de quedarnos con UN checkpoint,
tomamos los mejores N checkpoints al final del entrenamiento y promediamos
sus pesos. Esto encuentra el centroide del mínimo en el espacio de pérdida,
que matemáticamente corresponde a un mínimo más PLANO y más generalizable.

Referencia: Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima
and Better Generalization" — NIPS 2018.

Diferencia con PBT (Population Based Training, DeepMind 2017):
  PBT:  múltiples runs PARALELAS que compiten y se cruzan. Requiere cluster GPU.
  SWA:  un solo run, múltiples checkpoints del final, promedio matemático.
        Mismo efecto de "encontrar el mejor peso posible" sin costo extra.

Uso:
  python3 src/neural_manifold/swa_ensemble.py \
      --checkpoints data/03_models/unet_v3_ep3*.pth \
      --output      data/03_models/unet_v3_swa.pth \
      --val-split   data/05_totalsegmentator/dataset_split.json
"""

import os
import glob
import json
import argparse
import torch
import torchio as tio
import numpy as np
from torch.utils.data import DataLoader

from src.neural_manifold.unet_topology import UNet3D


# =====================================================================
# Utilidades de Validación
# =====================================================================

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


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin     = (pred.sigmoid() > threshold).float()
    intersection = (pred_bin * target).sum()
    denom        = pred_bin.sum() + target.sum()
    if denom == 0:
        return 1.0
    return (2. * intersection / denom).item()


def build_val_loader(split_path: str, patch_size: int = 128):
    """Construye el DataLoader de validación desde el split JSON."""
    with open(split_path) as f:
        split = json.load(f)
    
    subjects = []
    for p in split["validation"]:
        if os.path.exists(p["ct_path"]) and os.path.exists(p["mask_path"]):
            subjects.append(tio.Subject(
                ct=tio.ScalarImage(p["ct_path"]),
                label=tio.LabelMap(p["mask_path"])
            ))
    
    transform = tio.Compose([
        EnforceConsistentAffine(),
        EnsureMinShape((patch_size, patch_size, patch_size)),
    ])
    dataset = tio.SubjectsDataset(subjects, transform=transform)
    
    sampler = tio.data.LabelSampler(
        patch_size=patch_size,
        label_name='label',
        label_probabilities={0: 0.05, 1: 0.95}
    )
    queue = tio.Queue(dataset, max_length=50, samples_per_volume=2, sampler=sampler, num_workers=4)
    return DataLoader(queue, batch_size=1, num_workers=0), len(subjects)


def evaluate_checkpoint(model, val_loader, device) -> float:
    """Evalúa un modelo y retorna el Dice Score promedio."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in val_loader:
            X = batch['ct'][tio.DATA].float().to(device)
            Y = batch['label'][tio.DATA].float().to(device)
            pred = model(X)
            scores.append(dice_score(pred.cpu(), Y.cpu()))
    return float(np.mean(scores))


# =====================================================================
# Algoritmo Principal de SWA
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="SWA: Promedio de checkpoints para V3.1")
    parser.add_argument("--checkpoints",  nargs="+", required=True,
                        help="Lista de archivos .pth a evaluar y promediar")
    parser.add_argument("--output",       default="data/03_models/unet_v3_swa.pth",
                        help="Ruta de salida del modelo SWA")
    parser.add_argument("--val-split",    default="data/05_totalsegmentator/dataset_split.json",
                        help="Ruta al JSON de splits")
    parser.add_argument("--top-n",        type=int, default=5,
                        help="Cuántos checkpoints promediar (los mejores N)")
    parser.add_argument("--patch-size",   type=int, default=128)
    parser.add_argument("--skip-eval",    action="store_true",
                        help="Saltear la evaluación y promediar todos los checkpoints dados")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")

    checkpoints = sorted(args.checkpoints)
    print(f"[*] Checkpoints encontrados: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"    {cp}")

    # ---- FASE 1: Elitismo — Evaluar cada checkpoint y rankear ----
    if not args.skip_eval and os.path.exists(args.val_split):
        print(f"\n[*] FASE 1: Evaluando {len(checkpoints)} checkpoints en el validation set...")
        val_loader, n_val = build_val_loader(args.val_split, args.patch_size)
        print(f"    Pacientes de validación: {n_val}")

        scores = []
        for cp_path in checkpoints:
            model = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
            model.load_state_dict(torch.load(cp_path, map_location=device))
            score = evaluate_checkpoint(model, val_loader, device)
            scores.append((score, cp_path))
            print(f"    Dice Score: {score:.4f} ({score*100:.1f}%) — {os.path.basename(cp_path)}")

        # Ordenar por Dice Score descendente (elitismo)
        scores.sort(key=lambda x: x[0], reverse=True)

        print(f"\n[*] RANKING (Elitismo):")
        for rank, (score, path) in enumerate(scores, 1):
            marker = "★ ÉLITE" if rank <= args.top_n else ""
            print(f"    #{rank}: {score:.4f} ({score*100:.1f}%) — {os.path.basename(path)} {marker}")

        # Seleccionar los Top-N
        elite_checkpoints = [path for _, path in scores[:args.top_n]]
        best_single_dice  = scores[0][0]

        print(f"\n[*] Promediando los Top-{args.top_n} checkpoints...")
    else:
        print(f"\n[*] Modo skip-eval: promediando TODOS los checkpoints dados.")
        elite_checkpoints = checkpoints
        best_single_dice  = 0.0

    # ---- FASE 2: Promedio de Pesos (SWA) ----
    print(f"[*] FASE 2: Calculando promedio de pesos...")

    # Cargar el primero como base
    model_swa = UNet3D(in_channels=1, out_channels=1, base_features=32)
    base_state = torch.load(elite_checkpoints[0], map_location="cpu")
    swa_state  = {k: v.float().clone() for k, v in base_state.items()}

    # Acumular el resto
    for cp_path in elite_checkpoints[1:]:
        state = torch.load(cp_path, map_location="cpu")
        for k in swa_state:
            swa_state[k] += state[k].float()

    # Dividir por N (promedio)
    n = len(elite_checkpoints)
    for k in swa_state:
        swa_state[k] /= n

    # ---- FASE 3: Guardar y Evaluar el modelo SWA ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(swa_state, args.output)
    print(f"[✓] Modelo SWA guardado: {args.output}")

    if not args.skip_eval and os.path.exists(args.val_split):
        print(f"\n[*] FASE 3: Evaluando modelo SWA final...")
        model_swa.load_state_dict(swa_state)
        model_swa = model_swa.to(device)
        swa_dice = evaluate_checkpoint(model_swa, val_loader, device)

        print(f"\n{'='*50}")
        print(f" RESULTADO FINAL V3.1 (SWA)")
        print(f"{'='*50}")
        print(f"  Mejor checkpoint individual: {best_single_dice:.4f} ({best_single_dice*100:.1f}%)")
        print(f"  Modelo SWA (Top-{n}):        {swa_dice:.4f} ({swa_dice*100:.1f}%)")
        delta = swa_dice - best_single_dice
        print(f"  Mejora por SWA:              {delta:+.4f} ({delta*100:+.1f}%)")
        print(f"{'='*50}")

        if swa_dice > best_single_dice:
            print(f"\n[✓] SWA superó al mejor checkpoint individual.")
            print(f"    Este modelo es tu punto de partida para V4.")
        else:
            print(f"\n[!] SWA no superó al mejor individual.")
            print(f"    Usá el checkpoint #{1}: {elite_checkpoints[0]}")


if __name__ == "__main__":
    main()
