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
        # Inferir todo el volumen retornando mapa de calor (probabilidades 0 a 1)
        vol_pred = predict_volume_from_dicom(
            dicom_dir=dicom_test_dir, 
            model_path=model_path,
            device_str='cpu',
            return_probabilities=True
        )
        
        # Calcular masa de probabilidad (en lugar de conteo binario)
        masa_total = np.sum(vol_pred)
        print(f"  -> Masa de probabilidad de hueso en volumen 3D: {masa_total:.2f}")
        predicciones.append(vol_pred)
        
    # Detectar plano de corte según el tipo de dataset
    base_name = os.path.basename(os.path.normpath(dicom_test_dir))
    es_coronal = base_name in ["Fantoma_Pelvis", "Fantoma_CEMENER"]
    
    if es_coronal:
        plano = "Coronal (Y)"
        n_slices = X_raw.shape[1]
        get_slice_raw = lambda vol, idx: vol[:, idx, :]
        get_slice_pred = lambda vol, idx: vol[:, idx, :]
    else:
        plano = "Axial (Z)"
        n_slices = X_raw.shape[2]
        get_slice_raw = lambda vol, idx: vol[:, :, idx]
        get_slice_pred = lambda vol, idx: vol[:, :, idx]
    
    print(f"Plano de corte detectado: {plano}")
    
    # Autoseleccionar el mejor slice si no se proveyó uno
    if slice_index is None:
        masa_por_slice = [np.sum(get_slice_pred(predicciones[-1], s)) for s in range(n_slices)]
        slice_index = np.argmax(masa_por_slice)
        print(f"\nAutoseleccionado slice {slice_index}/{n_slices} basado en el pico de probabilidad de la red.")
            
    slice_original = get_slice_raw(X_raw, slice_index)
        
    # 3. Dibujar la progresión
    print("Graficando el Gradiente de Mejora Cualitativa...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Mostrar DICOM Original
    axes[0].imshow(slice_original, cmap='gray', vmin=-500, vmax=1500)
    axes[0].set_title(f"Tomografía Original\n({plano}, Slice {slice_index})")
    axes[0].axis('off')
    
    # Mostrar la evolución del Mapa de Calor
    for i, ep in enumerate(epocas):
        slice_pred = get_slice_pred(predicciones[i], slice_index)
        
        # Ocultar las probabilidades menores a 0.05 para limpiar el ruido del fondo
        slice_pred_masked = np.ma.masked_where(slice_pred < 0.05, slice_pred)
        
        # Masa local en este slice
        masa_local = np.sum(slice_pred)
        print(f"Época {ep}: Masa de probabilidad en slice {slice_index}: {masa_local:.2f}")
        
        # Superponer el mapa de calor (inferno) sobre el hueso
        axes[i+1].imshow(slice_original, cmap='gray', vmin=-500, vmax=1500)
        im = axes[i+1].imshow(slice_pred_masked, cmap='inferno', alpha=0.7, vmin=0, vmax=1)
        axes[i+1].set_title(f"Mapa Probabilidad Época {ep}\n(Suma P: {masa_local:.1f})")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    
    # 4. Definir nombre de salida según tipo de dataset
    if base_name.startswith("Paciente_"):
        output_filename = "progreso_entrenamiento.png"
    else:
        output_filename = f"progreso_entrenamiento_{base_name}.png"

    plt.savefig(output_filename, dpi=300)
    print(f"¡Imagen guardada como '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    # IMPORTANTE: Reemplaza esta ruta con una carpeta de DICOM de un paciente de prueba
    # que la IA no haya visto nunca (o uno del dataset para ver cómo aprende).
    DIR_PACIENTE_PRUEBA = "data/01_raw/Fantoma_CEMENER" 
    
    generar_comparacion_visual(DIR_PACIENTE_PRUEBA)
