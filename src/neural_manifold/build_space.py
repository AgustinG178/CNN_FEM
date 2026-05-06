import os
import random
import numpy as np
from src.tensor_pde.io_module import assemble_tensor_and_hu
from src.neural_manifold.patch_extractor import extract_isometric_subspaces

def build_training_manifold(
    raw_qct_dir: str, 
    raw_mask_dir: str, 
    output_dir: str, 
    patch_size: int = 64, 
    stride: int = 32,
    test_split_ratio: float = 0.15
) -> None:
    r"""
    Evalúa iterativamente la partición topológica \mathcal{P} sobre el conjunto 
    de datos \mathcal{D} y proyecta los tensores resultantes al sistema de archivos 
    para su inyección en el cargador iterativo (DataLoader).
    Garantiza la pureza del conjunto de prueba separando estocásticamente pacientes.
    """
    os.makedirs(os.path.join(output_dir, "tensors"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    patient_ids = [d for d in os.listdir(raw_qct_dir) 
                   if os.path.isdir(os.path.join(raw_qct_dir, d)) and 
                   (d.startswith("Paciente") or d.startswith("Fantoma"))]
    
    # --- Lógica de Partición Train/Test ---
    # Aseguramos que los Fantomas estén en test para validar el pipeline FEM después
    test_patients = [p for p in patient_ids if "Fantoma" in p]
    train_candidates = [p for p in patient_ids if "Fantoma" not in p]
    
    num_random_test = int(len(train_candidates) * test_split_ratio)
    random.seed(42) # Semilla fija para reproducibilidad
    random_test_patients = random.sample(train_candidates, num_random_test)
    test_patients.extend(random_test_patients)
    
    train_patients = [p for p in train_candidates if p not in random_test_patients]
    
    # Guardamos el registro de partición
    with open(os.path.join(output_dir, "test_patients_log.txt"), "w") as f:
        f.write("--- PACIENTES EXCLUIDOS PARA PRUEBA (TEST SET) ---\n")
        for tp in test_patients:
            f.write(f"{tp}\n")
    
    print(f"-> Total pacientes: {len(patient_ids)}")
    print(f"-> Asignados a Entrenamiento: {len(train_patients)}")
    print(f"-> Asignados a Prueba (Retenidos): {len(test_patients)}")

    global_patch_idx = 0
    
    for pid in train_patients:
        qct_path = os.path.join(raw_qct_dir, pid)
        mask_path = os.path.join(raw_mask_dir, pid + "_mask.npy") 
        
        if not os.path.exists(mask_path):
            print(f"[!] Máscara faltante para {pid}. Ejecuta auto_labeler.py primero. Omitiendo.")
            continue
            
        X_tensor = assemble_tensor_and_hu(qct_path)
        Y_tensor = np.load(mask_path)
        
        X_patches, Y_patches = extract_isometric_subspaces(
            X_tensor, Y_tensor, patch_size=patch_size, stride=stride
        )
        
        for i in range(X_patches.shape[0]):
            out_x_path = os.path.join(output_dir, "tensors", f"patch_{global_patch_idx:06d}.npy")
            out_y_path = os.path.join(output_dir, "masks", f"patch_{global_patch_idx:06d}.npy")
            
            np.save(out_x_path, X_patches[i])
            np.save(out_y_path, Y_patches[i])
            
            global_patch_idx += 1
            
    print(f"Partición finalizada. Dimensión del espacio de tensores de entrenamiento: {global_patch_idx}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    qct_in = os.path.join(BASE_DIR, "data", "01_raw")
    mask_in = os.path.join(BASE_DIR, "data", "01_ground_truth")
    patch_out = os.path.join(BASE_DIR, "data", "04_training_patches")
    
    build_training_manifold(qct_in, mask_in, patch_out)