import os
import pydicom
from datetime import datetime
import trimesh
import numpy as np

class OperadorBiomecanico:
    r"""
    Operador para la partición analítica de la variedad \partial \Omega_{femur}
    y generación dinámica del subespacio de almacenamiento del paciente.
    """
    def __init__(self, ruta_dicom: str, directorio_raiz: str):
        self.ruta_dicom = ruta_dicom
        self.directorio_raiz = directorio_raiz
        self.directorio_salida = self._computar_isomorfismo_directorio()

    def _computar_isomorfismo_directorio(self) -> str:
        r"""
        Evalúa el tensor de metadatos DICOM para instanciar un subespacio \mathbb{R}^3
        aislado para la matriz de datos del paciente actual.
        """
        archivo_muestra = next(
            os.path.join(raiz, f) for raiz, _, archivos in os.walk(self.ruta_dicom) 
            for f in archivos if f.endswith(('.dcm', ''))
        )
        metadatos = pydicom.dcmread(archivo_muestra, stop_before_pixels=True)
        id_paciente = str(getattr(metadatos, 'PatientID', 'Desconocido'))
        marca_temporal = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        nombre_directorio = f"Paciente_{id_paciente}_{marca_temporal}"
        ruta_salida = os.path.join(self.directorio_raiz, nombre_directorio)
        os.makedirs(ruta_salida, exist_ok=True)
        return ruta_salida

    def extraer_subdominios_femorales(self, ruta_stl: str) -> None:
        r"""
        Proyecta la variedad discreta sobre sus vectores propios \mathbf{v}_i
        e inyecta operadores de plano secante para aislar la topología.
        """
        malla_stl = trimesh.load_mesh(ruta_stl)
        
        covarianza = np.cov(malla_stl.vertices.T)
        valores_propios, vectores_propios = np.linalg.eigh(covarianza)
        
        eje_principal = vectores_propios[:, np.argmax(valores_propios)]
        eje_z_canonico = np.array([0, 0, 1])
        
        cos_theta = np.dot(eje_principal, eje_z_canonico) / (np.linalg.norm(eje_principal) * np.linalg.norm(eje_z_canonico))
        eje_rotacion = np.cross(eje_principal, eje_z_canonico)
        angulo_transformacion = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        matriz_afine = trimesh.transformations.rotation_matrix(angulo_transformacion, eje_rotacion)
        malla_stl.apply_transform(matriz_afine)
        
        z_min, z_max = malla_stl.bounds[:, 2]
        longitud_Lz = z_max - z_min
        
        cota_proximal = z_max - 0.22 * longitud_Lz
        cota_distal = z_min + 0.20 * longitud_Lz
        
        epifisis_proximal = malla_stl.slice_plane(plane_origin=[0, 0, cota_proximal], plane_normal=[0, 0, 1])
        diafisis_temporal = malla_stl.slice_plane(plane_origin=[0, 0, cota_proximal], plane_normal=[0, 0, -1])
        diafisis_estricta = diafisis_temporal.slice_plane(plane_origin=[0, 0, cota_distal], plane_normal=[0, 0, 1])
        epifisis_distal = malla_stl.slice_plane(plane_origin=[0, 0, cota_distal], plane_normal=[0, 0, -1])
        
        for sub_malla, nombre in zip(
            [epifisis_proximal, diafisis_estricta, epifisis_distal], 
            ["epifisis_proximal", "diafisis", "epifisis_distal"]
        ):
            if not sub_malla.is_empty:
                sub_malla.export(os.path.join(self.directorio_salida, f"subdominio_{nombre}.stl"))