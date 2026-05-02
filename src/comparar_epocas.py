import os
import matplotlib.pyplot as plt
import numpy as np
from src.neural_manifold.inference import predict_volume_from_dicom
from src.tensor_pde.io_module import assemble_tensor_and_hu

def generar_comparacion_visual(dicom_test_dir: str, slice_index: int = None):
    """
    Genera un panel de matplotlib comparando cómo la IA "esculpe" el hueso
    a lo largo de las primeras 4 épocas de entrenamiento.
    """
    print(f"Generando panel evolutivo para el paciente: {dicom_test_dir}")
    
    # 1. Cargar la imagen original (para usarla de fondo)
    X_raw = assemble_tensor_and_hu(dicom_test_dir)
    # Seleccionar el slice central del eje Z si no se especifica uno
    if slice_index is None:
        slice_index = X_raw.shape[2] // 2
        
    slice_original = X_raw[:, :, slice_index]
    
    # 2. Definir los modelos a evaluar
    epocas = [1, 2, 3, 4]
    predicciones = []
    
    for ep in epocas:
        model_path = f"data/03_models/unet_bone_topology_ep{ep}.pth"
        if not os.path.exists(model_path):
            print(f"[!] No se encontró localmente {model_path}. Por favor, descárgalo del clúster con WinSCP.")
            predicciones.append(np.zeros_like(X_raw))
            continue
            
        print(f"Evaluando Época {ep}...")
        # Inferir todo el volumen
        vol_pred = predict_volume_from_dicom(
            dicom_dir=dicom_test_dir, 
            model_path=model_path,
            device_str='cpu' # Aseguramos que corra en cualquier PC local
        )
        predicciones.append(vol_pred)
        
    # 3. Dibujar la progresión
    print("Graficando el Gradiente de Mejora Cualitativa...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Mostrar DICOM Original
    axes[0].imshow(slice_original, cmap='gray', vmin=-500, vmax=1500)
    axes[0].set_title("Tomografía Original")
    axes[0].axis('off')
    
    # Mostrar la evolución de las Máscaras
    for i, ep in enumerate(epocas):
        slice_pred = predicciones[i][:, :, slice_index]
        
        # Superponer la máscara roja sobre la tomografía
        axes[i+1].imshow(slice_original, cmap='gray', vmin=-500, vmax=1500)
        axes[i+1].imshow(slice_pred, cmap='Reds', alpha=0.5) # Máscara semitransparente
        axes[i+1].set_title(f"Predicción Época {ep}")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig("progreso_entrenamiento.png", dpi=300)
    print("¡Imagen guardada como 'progreso_entrenamiento.png'!")
    plt.show()

if __name__ == "__main__":
    # IMPORTANTE: Reemplaza esta ruta con una carpeta de DICOM de un paciente de prueba
    # que la IA no haya visto nunca (o uno del dataset para ver cómo aprende).
    DIR_PACIENTE_PRUEBA = "data/01_raw_dicom/Paciente_45" 
    
    generar_comparacion_visual(DIR_PACIENTE_PRUEBA)
