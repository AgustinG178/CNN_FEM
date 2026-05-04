import os
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
from src.neural_manifold.inference import predict_volume_from_dicom
from src.tensor_pde.io_module import assemble_tensor_and_hu

def generar_comparacion_visual(dicom_test_dir: str, slice_index: int = None):
    """
    Genera un panel de matplotlib comparando cómo la IA "esculpe" el hueso
    a lo largo de las épocas de entrenamiento. Se muestra en una grilla de 3 columnas.
    """
    print(f"Generando panel evolutivo para el paciente/fantoma: {dicom_test_dir}")
    
    # 1. Cargar la imagen original (para usarla de fondo)
    X_raw = assemble_tensor_and_hu(dicom_test_dir)
    
    # 2. Definir los modelos a evaluar buscando todos los .pth
    modelos = sorted(glob.glob("data/03_models/unet_bone_topology_ep*.pth"), 
                     key=lambda x: int(x.split('ep')[-1].split('.pth')[0]))
    
    if not modelos:
        print("[!] No se encontraron modelos en data/03_models/")
        return
        
    epocas = [int(m.split('ep')[-1].split('.pth')[0]) for m in modelos]
    print(f"Épocas encontradas: {epocas}")
    
    predicciones = []
    
    for ep, model_path in zip(epocas, modelos):
        print(f"Evaluando Época {ep}...")
        # Inferir todo el volumen retornando mapa de calor (probabilidades 0 a 1)
        vol_pred = predict_volume_from_dicom(
            dicom_dir=dicom_test_dir, 
            model_path=model_path,
            device_str='cpu',
            return_probabilities=True
        )
        
        # Calcular masa de probabilidad
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
    n_plots = 1 + len(epocas)  # 1 original + N épocas
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()
    
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
        
        # Superponer el mapa de calor (inferno) sobre el hueso
        axes[i+1].imshow(slice_original, cmap='gray', vmin=-500, vmax=1500)
        axes[i+1].imshow(slice_pred_masked, cmap='inferno', alpha=0.7, vmin=0, vmax=1)
        axes[i+1].set_title(f"Época {ep}\n(Suma P: {masa_local:.1f})")
        axes[i+1].axis('off')
        
    # Ocultar ejes no utilizados
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')
        
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
    # Usamos un fantoma para aprovechar los cortes coronales (ideal para ver la pelvis entera)
    DIR_PACIENTE_PRUEBA = "data/01_raw/Paciente_22" 
    
    generar_comparacion_visual(DIR_PACIENTE_PRUEBA)
