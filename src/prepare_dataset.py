import os
from src.neural_manifold.auto_labeler import generate_ground_truth_for_all_patients
from src.neural_manifold.build_space import build_training_manifold

def prepare_dataset_pipeline():
    """
    Orquesta la autogeneración de etiquetas y la subsecuente partición 
    del espacio volumétrico para el entrenamiento de la red neuronal.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.join(BASE_DIR, "data", "01_raw")
    GT_DIR = os.path.join(BASE_DIR, "data", "01_ground_truth")
    PATCH_OUT = os.path.join(BASE_DIR, "data", "04_training_patches")
    
    print("=====================================================")
    print("   INICIANDO PREPARACIÓN AUTOMÁTICA DEL DATASET")
    print("=====================================================")
    
    print("\n[PASO 1/2] Autogenerando Etiquetas con IA preentrenada (Ground Truth)...")
    generate_ground_truth_for_all_patients(RAW_DIR, GT_DIR)
    
    print("\n[PASO 2/2] Particionando espacio en Parches 3D y separando Train/Test...")
    build_training_manifold(
        raw_qct_dir=RAW_DIR, 
        raw_mask_dir=GT_DIR, 
        output_dir=PATCH_OUT, 
        patch_size=64, 
        stride=32,
        test_split_ratio=0.15
    )
    
    print("\n=====================================================")
    print(" ¡PROCESO COMPLETADO EXITOSAMENTE!")
    print(f" -> Parches de entrenamiento generados en: {PATCH_OUT}")
    print(" -> Ya puedes subir esa carpeta a tu clúster para entrenar.")
    print("=====================================================")

if __name__ == "__main__":
    prepare_dataset_pipeline()
