import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

# Agregar la raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_visuals(num_images=5):
    patch_dir = "data/04_training_patches"
    tensor_dir = os.path.join(patch_dir, "tensors")
    mask_dir = os.path.join(patch_dir, "masks")
    output_dir = "assets_informe/visuals_check"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(tensor_dir):
        print(f"[!] Error: No se encuentra la carpeta de parches en {tensor_dir}")
        return

    all_patches = [f for f in os.listdir(tensor_dir) if f.endswith(".npy")]
    random.shuffle(all_patches)
    
    count = 0
    print(f"-> Generando {num_images} comparativas aleatorias...")
    
    for p_name in all_patches:
        if count >= num_images:
            break
            
        # Cargar tensor y máscara
        X = np.load(os.path.join(tensor_dir, p_name))
        Y = np.load(os.path.join(mask_dir, p_name))
        
        # Solo nos interesan parches con hueso para el informe
        if np.sum(Y) < 500: # Filtro de cardinalidad mínima
            continue
            
        mid = X.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.suptitle(f"Verificación de Alineación Espacial - {p_name}", fontsize=14)
        
        # Vistas: Axial, Sagital, Coronal
        vistas = [
            ("Axial (XY)", X[:, :, mid], Y[:, :, mid]),
            ("Sagital (XZ)", X[:, mid, :], Y[:, mid, :]),
            ("Coronal (YZ)", X[mid, :, :], Y[mid, :, :])
        ]
        
        for i, (title, img, mask) in enumerate(vistas):
            axes[i].imshow(img, cmap='gray')
            # Superponer máscara con transparencia (color rojo/fucsia)
            mask_masked = np.ma.masked_where(mask == 0, mask)
            axes[i].imshow(mask_masked, cmap='autumn', alpha=0.5)
            axes[i].set_title(title)
            axes[i].axis('off')
            
        out_path = os.path.join(output_dir, f"check_{p_name.replace('.npy', '.png')}")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        
        print(f"   [+] Generada: {out_path}")
        count += 1

    print(f"\n[✓] Proceso finalizado. Las imágenes están en: {output_dir}")

if __name__ == "__main__":
    create_visuals(num_images=8)
