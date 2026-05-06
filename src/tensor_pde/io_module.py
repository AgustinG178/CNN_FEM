import pydicom
import numpy as np
import os


def extract_affine_matrix(dicom_path: str) -> np.ndarray:
    """
    Computa la matriz de transformación afín T (4x4) desde el espacio 
    discreto del vóxel (p,q,r) al espacio métrico euclidiano LPS (x,y,z).
    """
    dataset = pydicom.dcmread(dicom_path)
    
    #Extracción de cosenos directores (Vectores ortonormales F y D)
    F_x, F_y, F_z, D_x, D_y, D_z = dataset.ImageOrientationPatient
    F = np.array([F_x, F_y, F_z])
    D = np.array([D_x, D_y, D_z])
    
    #Cómputo del vector normal ortogonal al plano de corte
    N = np.cross(F, D)
    
    #Métrica del espacio (Escalado intrapíxel)
    delta_r, delta_c = dataset.PixelSpacing
    
    # Métrica interplanar (diferencial en el eje normal)
    if 'SpacingBetweenSlices' in dataset:
        delta_s = dataset.SpacingBetweenSlices
    elif 'SliceThickness' in dataset:
        delta_s = dataset.SliceThickness
    else:
        raise ValueError("El tensor carece de métrica interplanar explícita.")
        
    T_x, T_y, T_z = dataset.ImagePositionPatient
    
    T = np.array([
        [F[0] * delta_r, D[0] * delta_c, N[0] * delta_s, T_x],
        [F[1] * delta_r, D[1] * delta_c, N[1] * delta_s, T_y],
        [F[2] * delta_r, D[2] * delta_c, N[2] * delta_s, T_z],
        [0.0,            0.0,            0.0,            1.0]
    ])
    
    return T


def assemble_tensor_and_hu(directory_path: str) -> np.ndarray:
    """
    Ensambla el tensor discreto tridimensional X in R^{Nx x Ny x Nz} y mapea 
    los valores de los píxeles al campo escalar de Unidades Hounsfield (HU),
    garantizando homogeneidad dimensional y filtrando topogramas aberrantes.
    Busca archivos DICOM recursivamente.
    """
    dicom_files = []
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            if f.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, f))
    
    slices = []
    for f in dicom_files:
        ds = pydicom.dcmread(f)
        
        image_type = getattr(ds, 'ImageType', [])
        is_localizer = 'LOCALIZER' in image_type
        
        if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'ImageOrientationPatient') and not is_localizer:
            slices.append(ds)
            
    if not slices:
        raise ValueError("El directorio carece de tensores axiales válidos.")

    valid_rows = max([s.Rows for s in slices])
    valid_cols = max([s.Columns for s in slices])
    
    homogeneous_slices = [s for s in slices if s.Rows == valid_rows and s.Columns == valid_cols]
    
    homogeneous_slices.sort(key=lambda s: s.ImagePositionPatient[2], reverse=False)
    
    N_x = homogeneous_slices[0].Rows
    N_y = homogeneous_slices[0].Columns
    N_z = len(homogeneous_slices)
    
    X_tensor = np.zeros((N_x, N_y, N_z), dtype=np.float32)
    
    for k, s in enumerate(homogeneous_slices):
        img_2d = s.pixel_array.astype(np.float32)
        m = getattr(s, 'RescaleSlope', 1.0)
        b = getattr(s, 'RescaleIntercept', 0.0)
        
        X_tensor[:, :, k] = (img_2d * m) + b
        
    return X_tensor