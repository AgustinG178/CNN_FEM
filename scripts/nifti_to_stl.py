"""
nifti_to_stl.py
---------------
Genera mallas STL directamente desde máscaras NIfTI (.nii.gz).
Funciona tanto con:
  - Máscaras PREDICHAS por la red (pred_mask.nii.gz)
  - Máscaras de REFERENCIA de TotalSegmentator (bone_mask.nii.gz)

Pipeline:
  1. Cargar máscara NIfTI binaria
  2. Extraer espaciado real del header (mm por vóxel)
  3. Marching Cubes con espaciado real → vértices en mm
  4. Limpiar islas pequeñas (top 5 componentes anatómicas)
  5. Suavizado Taubin (preserva volumen, no encoge)
  6. Guardar STL

Uso standalone:
  python3 scripts/nifti_to_stl.py \
      --mask  data/02_processed/test_nifti_ep10/s0001_pred_mask.nii.gz \
      --output data/02_processed/test_nifti_ep10/s0001_pred.stl

Uso como módulo:
  from scripts.nifti_to_stl import mask_nifti_to_stl
  mesh = mask_nifti_to_stl("pred_mask.nii.gz", "output.stl")
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import trimesh
from skimage.measure import marching_cubes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_voxel_spacing(nifti_img) -> tuple:
    """
    Extrae el espaciado real (mm/vóxel) desde el header NIfTI.
    El espaciado es crítico para que las medidas del STL estén en mm reales
    y no en índices de vóxel. Un STL con vóxeles de 1×1×1 mm es inútil
    para COMSOL si la CT real tiene vóxeles de 0.7×0.7×1.5 mm.
    """
    header  = nifti_img.header
    zooms   = header.get_zooms()          # (sx, sy, sz) en mm
    spacing = tuple(float(z) for z in zooms[:3])
    return spacing


def clean_mesh(mesh: trimesh.Trimesh, n_components: int = 5) -> trimesh.Trimesh:
    """
    Conserva solo las N componentes conexas más grandes.
    Anatomía: pelvis izq/der, sacro, fémur izq, fémur der = 5 estructuras.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= n_components:
        return mesh
    sorted_components = sorted(components, key=lambda c: c.area, reverse=True)
    top = sorted_components[:n_components]
    return trimesh.util.concatenate(top)


def taubin_smooth(mesh: trimesh.Trimesh,
                  iterations: int = 20,
                  lambda_: float = 0.5,
                  mu_: float = -0.53) -> trimesh.Trimesh:
    """
    Suavizado de Taubin: alterna entre contracción (λ) y expansión (μ).
    A diferencia del suavizado laplaciano puro, NO encoge el volumen.
    Esencial para FEM: una malla encogida cambia las propiedades mecánicas.

    Referencia: Taubin (1995) "A signal processing approach to fair surface design"
    """
    verts = mesh.vertices.copy()
    adj   = mesh.vertex_adjacency_graph  # Grafo de adyacencia

    for _ in range(iterations):
        # Paso 1: Contracción (λ)
        delta = np.zeros_like(verts)
        for v_idx in range(len(verts)):
            neighbors = list(adj.neighbors(v_idx))
            if neighbors:
                delta[v_idx] = np.mean(verts[neighbors], axis=0) - verts[v_idx]
        verts += lambda_ * delta

        # Paso 2: Expansión anti-encogimiento (μ)
        delta = np.zeros_like(verts)
        for v_idx in range(len(verts)):
            neighbors = list(adj.neighbors(v_idx))
            if neighbors:
                delta[v_idx] = np.mean(verts[neighbors], axis=0) - verts[v_idx]
        verts += mu_ * delta

    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def mask_nifti_to_stl(
    mask_path: str,
    output_stl: str,
    n_components: int = 5,
    smooth_iterations: int = 15,
    step_size: int = 1
) -> trimesh.Trimesh:
    """
    Pipeline completo: NIfTI binaria → STL listo para COMSOL.

    Args:
        mask_path:         Ruta a la máscara binaria .nii.gz
        output_stl:        Ruta de salida del .stl
        n_components:      Máximo de componentes anatómicas a conservar
        smooth_iterations: Iteraciones de Taubin (más = más suave)
        step_size:         Resolución de Marching Cubes (1 = máxima calidad)

    Returns:
        mesh: Objeto trimesh de la malla generada
    """
    print(f"[*] Cargando máscara: {os.path.basename(mask_path)}")
    img      = nib.load(mask_path)
    mask     = img.get_fdata() > 0.5
    spacing  = get_voxel_spacing(img)

    print(f"    Dimensiones: {mask.shape} | Espaciado: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm")
    print(f"    Vóxeles de hueso: {mask.sum():,}")

    if mask.sum() == 0:
        print(f"    [!] Máscara vacía, no se puede generar STL")
        return None

    # 1. Marching Cubes con espaciado real
    print(f"[*] Ejecutando Marching Cubes (step_size={step_size})...")
    verts, faces, normals, _ = marching_cubes(
        volume=mask.astype(np.float32),
        level=0.5,
        spacing=spacing,
        step_size=step_size
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    print(f"    Vértices: {len(verts):,} | Caras: {len(faces):,}")

    # 2. Limpiar islas pequeñas
    print(f"[*] Conservando Top-{n_components} componentes anatómicas...")
    mesh = clean_mesh(mesh, n_components)
    print(f"    Vértices post-limpieza: {len(mesh.vertices):,}")

    # 3. Suavizado Taubin
    print(f"[*] Aplicando suavizado Taubin ({smooth_iterations} iteraciones)...")
    mesh = taubin_smooth(mesh, iterations=smooth_iterations)

    # 4. Guardar
    os.makedirs(os.path.dirname(os.path.abspath(output_stl)), exist_ok=True)
    mesh.export(output_stl)
    size_mb = os.path.getsize(output_stl) / 1e6
    print(f"[✓] STL guardado: {output_stl} ({size_mb:.1f} MB)")

    return mesh


def main():
    parser = argparse.ArgumentParser(description="NIfTI → STL para FEM")
    parser.add_argument("--mask",       required=True, help="Ruta a la máscara .nii.gz")
    parser.add_argument("--output",     required=True, help="Ruta de salida .stl")
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--smooth",     type=int, default=15)
    parser.add_argument("--step-size",  type=int, default=1)
    args = parser.parse_args()

    mask_nifti_to_stl(
        mask_path=args.mask,
        output_stl=args.output,
        n_components=args.components,
        smooth_iterations=args.smooth,
        step_size=args.step_size
    )


if __name__ == "__main__":
    main()
