import numpy as np
from itertools import product

def compute_padding_bounds(tensor_shape: tuple, patch_shape: tuple) -> list:
    r"""
    Evalúa el diferencial métrico en las fronteras del dominio para garantizar 
    que \mathbf{X} sea estrictamente divisible por las dimensiones del parche.
    Retorna la tupla de límites de padding [(0, p_x), (0, p_y), (0, p_z)].
    """
    pad_bounds = []
    for dim_size, p_size in zip(tensor_shape, patch_shape):
        remainder = dim_size % p_size
        if remainder == 0:
            pad_bounds.append((0, 0))
        else:
            pad_bounds.append((0, p_size - remainder))
    return pad_bounds

def extract_isometric_subspaces(
    X_tensor: np.ndarray, 
    Y_tensor: np.ndarray = None, 
    patch_size: int = 64, 
    stride: int = 32
                                ):
    
    r"""
    Aplica el operador de partición \mathcal{P}: \mathbb{R}^{N_x \times N_y \times N_z} \to \mathbb{R}^{K \times D \times H \times W}
    mediante una ventana deslizante con solapamiento espacial.
    """
    
    patch_shape = (patch_size, patch_size, patch_size)
    pad_bounds = compute_padding_bounds(X_tensor.shape, patch_shape)
    
    # Inyección de frontera nula (Padding)
    X_padded = np.pad(X_tensor, pad_bounds, mode='constant', constant_values=-1000.0) # -1000 HU = Aire
    if Y_tensor is not None:
        Y_padded = np.pad(Y_tensor, pad_bounds, mode='constant', constant_values=0.0)
    
    shape = X_padded.shape
    x_coords = range(0, shape[0] - patch_size + 1, stride)
    y_coords = range(0, shape[1] - patch_size + 1, stride)
    z_coords = range(0, shape[2] - patch_size + 1, stride)
    
    X_patches = []
    Y_patches = []
    
    for x, y, z in product(x_coords, y_coords, z_coords):
        X_sub = X_padded[x:x+patch_size, y:y+patch_size, z:z+patch_size]
        
        if Y_tensor is not None:
            Y_sub = Y_padded[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            
            # --- FILTRO TOPOLÓGICO (Negative Sampling) ---
            # Si el parche no contiene ni un solo vóxel de hueso, lo descartamos
            # en un 95% de los casos. Conservamos el 5% para que la red sepa 
            # cómo se ve el fondo vacío y no genere falsos positivos.
            if np.sum(Y_sub > 0) == 0:
                if np.random.rand() > 0.05:
                    continue # Descartar parche vacío
                    
            Y_patches.append(Y_sub)
            
        X_patches.append(X_sub)
            
    X_out = np.stack(X_patches)
    Y_out = np.stack(Y_patches) if Y_tensor is not None else None
    
    return X_out, Y_out