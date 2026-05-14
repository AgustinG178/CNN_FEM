"""
test_inference_nifti.py
-----------------------
Inferencia sobre tomografías NIfTI (.nii.gz) usando el modelo V3.
Usa Sliding Window para procesar volúmenes completos de cuerpo entero
que son más grandes que el parche de entrenamiento (128³).

Uso desde el clúster:
  python3 scripts/test_inference_nifti.py \
      --patient data/05_totalsegmentator/processed/s0001 \
      --checkpoint data/03_models/unet_v3_ep10.pth \
      --output-dir data/02_processed/test_nifti_ep10
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_manifold.unet_topology import UNet3D


# =====================================================================
# INFERENCIA POR VENTANA DESLIZANTE (Sliding Window)
# =====================================================================
def sliding_window_inference(
    volume: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 128,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Infiere la segmentación de un volumen 3D completo usando ventana deslizante.
    Promedia las predicciones en las zonas de solapamiento (overlap) para
    evitar artefactos en los bordes de los parches.

    Args:
        volume:     Volumen 3D normalizado de forma (D, H, W)
        model:      Modelo UNet3D ya cargado
        patch_size: Tamaño del cubo de inferencia (igual al de entrenamiento)
        overlap:    Fracción de solapamiento entre parches (0.5 = 50%)

    Returns:
        prob_map: Mapa de probabilidades (D, H, W) con valores en [0, 1]
    """
    D, H, W = volume.shape
    step = int(patch_size * (1 - overlap))

    # Acumuladores: suma de predicciones y conteo de veces que se predijo cada vóxel
    pred_sum   = np.zeros((D, H, W), dtype=np.float32)
    pred_count = np.zeros((D, H, W), dtype=np.float32)

    model.eval()

    with torch.no_grad():
        # Iterar en las 3 dimensiones
        for z in range(0, D - patch_size + 1, step):
            for y in range(0, H - patch_size + 1, step):
                for x in range(0, W - patch_size + 1, step):
                    # Extraer parche
                    patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    patch_tensor = torch.from_numpy(patch).float()
                    patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]

                    # Inferencia
                    pred = model(patch_tensor)
                    prob = torch.sigmoid(pred).squeeze().cpu().numpy()

                    # Acumular con ventana gaussiana (suaviza las costuras)
                    pred_sum[z:z+patch_size, y:y+patch_size, x:x+patch_size]   += prob
                    pred_count[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1.0

    # Evitar división por cero en bordes
    pred_count = np.maximum(pred_count, 1.0)
    prob_map   = pred_sum / pred_count

    return prob_map


def prepare_ct(ct_data: np.ndarray) -> np.ndarray:
    """
    Prepara el CT exactamente igual que Torchio durante el entrenamiento:
    NO normaliza. Torchio carga los valores HU crudos como float tensor.
    Solo recortamos outliers extremos para estabilidad numérica.
    Rango típico: aire=-1000, tejido blando=0-100, hueso=300-1500 HU.
    """
    # Mismo rango que Torchio usa internamente: clip extremo pero SIN normalizar
    return np.clip(ct_data, -1024.0, 3000.0).astype(np.float32)


def otsu_threshold(prob_map: np.ndarray) -> float:
    """
    Calcula el umbral óptimo de Otsu sobre el mapa de probabilidades.
    Más robusto que un umbral fijo de 0.5 cuando el modelo no está calibrado.
    """
    # Solo considerar vóxeles con probabilidad > 0.05 (ignorar fondo absoluto)
    probs_flat = prob_map[prob_map > 0.05].flatten()
    if len(probs_flat) == 0:
        return 0.5
    # Histograma de 256 bins para Otsu
    hist, bin_edges = np.histogram(probs_flat, bins=256, range=(0, 1))
    hist = hist.astype(float) / hist.sum()
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Criterio de Otsu: maximizar varianza entre clases
    best_thresh, best_var = 0.5, 0.0
    for i in range(1, len(hist)):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:i] * bins[:i]).sum() / w0
        m1 = (hist[i:] * bins[i:]).sum() / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_thresh = var, bins[i]
    return best_thresh


def keep_largest_components(mask: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    Filtra el mapa binario conservando solo las N componentes conexas más grandes.
    Anatómicamente, los huesos pélvicos son las estructuras más grandes.
    Elimina los falsos positivos dispersos (ruido de predicción).
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    # Calcular tamaño de cada componente
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    # Quedarse con las N más grandes
    top_n = np.argsort(sizes)[::-1][:n_components]
    clean_mask = np.zeros_like(mask)
    for idx in top_n:
        clean_mask[labeled == (idx + 1)] = 1
    return clean_mask


def pad_to_multiple(volume: np.ndarray, patch_size: int) -> tuple:
    """
    Rellena el volumen para que cada dimensión sea múltiplo del patch_size.
    Retorna el volumen rellenado y el padding aplicado.
    """
    D, H, W = volume.shape
    pad_d = (patch_size - D % patch_size) % patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    padded = np.pad(volume,
                    ((0, pad_d), (0, pad_h), (0, pad_w)),
                    mode='constant', constant_values=0)
    return padded, (pad_d, pad_h, pad_w)


def main():
    parser = argparse.ArgumentParser(description="Inferencia NIfTI — BoneFlow V3")
    parser.add_argument("--patient",     required=True,
                        help="Directorio del paciente procesado (con ct.nii.gz)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Ruta al checkpoint .pth")
    parser.add_argument("--output-dir",  default="data/02_processed/test_nifti",
                        help="Directorio de salida")
    parser.add_argument("--patch-size",  type=int, default=128)
    parser.add_argument("--overlap",     type=float, default=0.5,
                        help="Fracción de solapamiento entre parches (0.5 recomendado)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Umbral de probabilidad para binarizar la máscara")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f" INFERENCIA NIFTI — BONEFLOW V3")
    print(f"{'='*55}")
    print(f"[*] Dispositivo: {device}")
    print(f"[*] Checkpoint:  {args.checkpoint}")
    print(f"[*] Paciente:    {args.patient}")

    # 1. Cargar CT
    ct_path = os.path.join(args.patient, "ct.nii.gz")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"No se encontró {ct_path}")

    ct_img  = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    affine  = ct_img.affine
    print(f"[*] Dimensiones CT: {ct_data.shape}")

    # 2. Cargar máscara de referencia (para calcular Dice)
    mask_path = os.path.join(args.patient, "bone_mask.nii.gz")
    has_mask  = os.path.exists(mask_path)
    if has_mask:
        mask_data = nib.load(mask_path).get_fdata() > 0
        print(f"[*] Máscara de referencia encontrada (calcularemos Dice real)")

    # 3. Pre-procesamiento CORREGIDO: mismo que Torchio durante entrenamiento
    print("[*] Preparando CT (HU crudos, sin normalizar — igual que Torchio)...")
    ct_norm = prepare_ct(ct_data)

    # Rellenar para que el sliding window cubra todo el volumen
    ct_padded, padding = pad_to_multiple(ct_norm, args.patch_size)
    print(f"[*] Volumen rellenado: {ct_padded.shape}")

    # 4. Cargar modelo
    print(f"[*] Cargando modelo...")
    model = UNet3D(in_channels=1, out_channels=1, base_features=32).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"[✓] Modelo cargado correctamente")

    # 5. Inferencia por sliding window
    print(f"[*] Ejecutando Sliding Window Inference "
          f"(patch={args.patch_size}³, overlap={args.overlap*100:.0f}%)...")
    print(f"    (Esto puede tardar varios minutos en CPU...)")

    prob_map_padded = sliding_window_inference(
        ct_padded, model, device, args.patch_size, args.overlap
    )

    # Recortar el padding
    D, H, W = ct_data.shape
    prob_map = prob_map_padded[:D, :H, :W]
    # 6. Binarización adaptativa con Otsu
    otsu_t = otsu_threshold(prob_map)
    print(f"[*] Umbral Otsu calculado: {otsu_t:.4f} (fijo solicitado: {args.threshold})")
    # Usamos el mayor entre Otsu y el umbral solicitado (más conservador)
    threshold_used = max(otsu_t, args.threshold)
    print(f"[*] Umbral final aplicado: {threshold_used:.4f}")
    mask_pred = (prob_map > threshold_used).astype(np.uint8)

    # Filtrar componentes pequeñas (ruido de predicción)
    print("[*] Filtrando componentes conexas (conservando las 5 más grandes)...")
    mask_pred = keep_largest_components(mask_pred, n_components=5)

    print(f"[✓] Inferencia completada")
    print(f"    Vóxeles de hueso predichos: {mask_pred.sum():,}")
    print(f"    Fracción del volumen:       {mask_pred.mean()*100:.2f}%")

    # 6. Calcular Dice si hay máscara de referencia
    if has_mask:
        intersection = (mask_pred * mask_data).sum()
        dice = 2.0 * intersection / (mask_pred.sum() + mask_data.sum() + 1e-6)
        print(f"\n[★] DICE SCORE REAL: {dice:.4f} ({dice*100:.1f}%)")
        print(f"    (Vóxeles reales de hueso: {mask_data.sum():,})")

    # 7. Guardar outputs
    patient_id = os.path.basename(args.patient)

    # Guardar mapa de probabilidades
    prob_nifti = nib.Nifti1Image(prob_map.astype(np.float32), affine)
    prob_path  = os.path.join(args.output_dir, f"{patient_id}_prob_map.nii.gz")
    nib.save(prob_nifti, prob_path)
    print(f"\n[✓] Mapa de probabilidades guardado: {prob_path}")

    # Guardar máscara binaria
    mask_nifti = nib.Nifti1Image(mask_pred, affine)
    mask_path_out = os.path.join(args.output_dir, f"{patient_id}_pred_mask.nii.gz")
    nib.save(mask_nifti, mask_path_out)
    print(f"[✓] Máscara predicha guardada:      {mask_path_out}")

    # 8. Generar malla STL con pipeline NIfTI nativo
    try:
        from scripts.nifti_to_stl import mask_nifti_to_stl
        patient_id = os.path.basename(args.patient)
        stl_out    = os.path.join(args.output_dir, f"{patient_id}_pred.stl")
        print(f"\n[*] Generando malla 3D STL...")
        mask_nifti_to_stl(mask_path_out, stl_out)
    except Exception as e:
        print(f"[!] Generación STL omitida: {e}")

    print(f"\n{'='*55}")
    print(f" INFERENCIA COMPLETADA — {patient_id}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
