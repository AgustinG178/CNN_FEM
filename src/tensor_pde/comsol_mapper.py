import os
import numpy as np
import trimesh

def map_hu_to_young_modulus(hu_array: np.ndarray) -> np.ndarray:
    r"""
    Evalúa la biyección empírica \rho(\mathbf{x}) y E(\mathbf{x}) sobre el campo escalar HU(\mathbf{x}),
    retornando el tensor espacial del módulo de elasticidad longitudinal en MPa.
    """
    rho = (1.067e-3 * hu_array) + 0.131
    rho = np.clip(rho, a_min=0.01, a_max=None)
    E = 3790.0 * (rho ** 3.0)
    
    return E

def compute_stiffness_tensor(E: float, nu: float = 0.3) -> np.ndarray:
    r"""
    Ensambla la matriz de rigidez isótropa \mathbf{C} \in \mathbb{R}^{6 \times 6} bajo la formulación 
    de Lamé para un diferencial espacial volumétrico.
    """
    lambda_lame = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_lame = E / (2.0 * (1.0 + nu))
    
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 0] = C[1, 1] = C[2, 2] = lambda_lame + 2.0 * mu_lame
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lambda_lame
    C[3, 3] = C[4, 4] = C[5, 5] = mu_lame
    
    return C

def export_heterogeneous_field(X_tensor: np.ndarray, T_matrix: np.ndarray, output_path: str, tau: float = 100.0):
    r"""
    Ensambla la matriz \mathbf{M} \in \mathbb{R}^{N \times 4} y aplica un operador de decimación espacial.
    Previene la divergencia de memoria en la rutina de mallado volumétrico de COMSOL.
    """
    indices = np.argwhere(X_tensor >= tau)
    
    i = indices[:, 0]
    j = indices[:, 1]
    k = indices[:, 2]
    ones = np.ones_like(i)
    
    P_vox = np.vstack((j, i, k, ones))
    P_phys = T_matrix @ P_vox
    
    hu_values = X_tensor[i, j, k]
    E_values = map_hu_to_young_modulus(hu_values)
    
    M = np.column_stack((P_phys[0, :], P_phys[1, :], P_phys[2, :], E_values))
    
    r"""
    Purga estricta del subespacio nulo o divergente.
    """
    M = M[~np.isnan(M).any(axis=1)]
    M = M[~np.isinf(M).any(axis=1)]
    
    r"""
    Decimación Espacial mediante submuestreo iterativo.
    Preserva la macro-heterogeneidad del campo E(\mathbf{x}) estabilizando la EDP.
    """
    stride = 25
    M_decimated = M[::stride, :]
    
    np.savetxt(output_path, M_decimated, fmt='%.4f', delimiter='\t')
    print(f"\nMatriz M decimada a {M_decimated.shape[0]} vectores para interpolación estable en Delaunay.")

def export_comsol_selection(mesh_path: str, output_csv: str, mode: str = 'acetabulum', base_radius: float = 35.0):
    r"""
    Identifica sub-variedades \Gamma \subset \partial \Omega y genera vectores de coordenadas.
    Implementa convergencia adaptativa para asegurar la cardinalidad de los nodos de carga.
    """
    if not os.path.exists(mesh_path):
        return

    mesh = trimesh.load(mesh_path)
    vertices = mesh.vertices

    if mode == 'acetabulum':
        center_ref = mesh.centroid
        radius = base_radius
        mask = np.linalg.norm(vertices - center_ref, axis=1) <= radius
        
        r"""
        Bucle heurístico para evadir singularidades topológicas (|V| \ll 150)
        debido a desviaciones en el cálculo analítico del centroide ilíaco.
        """
        while np.sum(mask) < 150 and radius < 80.0:
            radius += 5.0
            mask = np.linalg.norm(vertices - center_ref, axis=1) <= radius

    elif mode == 'sacrum_fix':
        z_max = vertices[:, 2].max()
        mask = vertices[:, 2] >= (z_max - 5.0)
        
    else:
        raise ValueError("Modo analítico no definido.")

    selection_coords = vertices[mask]
    np.savetxt(output_csv, selection_coords, delimiter=',', header="x, y, z", comments='')
    print(f"-> Selección '{mode}' exportada: {len(selection_coords)} puntos en {output_csv}")

def map_all_selections(processed_dir: str, selections_dir: str):
    r"""
    Orquesta la parametrización de las fronteras de Neumann y Dirichlet
    sobre el espacio de procesados.
    """
    os.makedirs(selections_dir, exist_ok=True)
    
    sacrum_path = os.path.join(processed_dir, "dominio_sacrum.stl")
    export_comsol_selection(sacrum_path, os.path.join(selections_dir, "fix_sacrum.csv"), mode='sacrum_fix')
    
    for side in ['right', 'left']:
        hip_path = os.path.join(processed_dir, f"dominio_hip_{side}.stl")
        export_comsol_selection(hip_path, os.path.join(selections_dir, f"load_acetabulum_{side}.csv"), mode='acetabulum')