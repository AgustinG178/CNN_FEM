import numpy as np
import nibabel as nib

def generate_comsol_material_field(nifti_path: str, output_txt: str, a: float = 0.0, b: float = 0.001, c: float = 15000.0, d: int = 2):
    r"""
    Genera un campo escalar heterogéneo E(x,y,z) compatible con COMSOL Multiphysics.
    Utiliza una biyección bilineal para transformar Unidades Hounsfield (HU) en 
    densidad aparente \rho, y aplica la Ley de Wolff (relación potencial) para 
    derivar el Módulo de Young local.
    
    Parámetros físicos por defecto:
    - a, b: Coeficientes de calibración HU -> \rho (g/cm^3)
    - c, d: Parámetros de rigidez ósea (E = c * \rho^d)
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    
    #Mapeo a Densidad Aparente y Módulo elástico
    rho_app = a + b * data
    rho_app = np.maximum(rho_app, 0.0) # Restricción física: rho >= 0
    E_field = c * (rho_app**d)
    
    z, y, x = np.where(E_field > 0) # Solo puntos con masa
    indices = np.stack((x, y, z, np.ones_like(x)), axis=-1)
    coords_phys = (affine @ indices.T).T[:, :3]
    
    values = E_field[z, y, x]
    export_data = np.column_stack((coords_phys, values))
    
    print(f"-> Exportando {len(export_data)} puntos de material...")
    np.savetxt(output_txt, export_data, header="x y z E", comments='')

# generate_comsol_material_field("ct_volume.nii.gz", "propiedades_material.txt")