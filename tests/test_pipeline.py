"""
Tests Unitarios del Pipeline Biomecánico.

Ejecutar con:
    python -m pytest tests/ -v

Estos tests utilizan tensores sintéticos y NO requieren datos DICOM reales,
modelos entrenados ni acceso al clúster HPC. Se ejecutan en < 5 segundos.
"""
import pytest
import numpy as np
import torch
import os
import tempfile


# ============================================================
# TEST 1: Simetría dimensional de la UNet3D
# ============================================================
class TestUNet3DArchitecture:
    """
    Verifica que la arquitectura de la red neuronal preserva la dimensionalidad
    espacial del tensor de entrada. Si alguien modifica las capas de contracción
    o expansión y rompe la simetría, este test lo detecta instantáneamente.
    """

    def test_output_shape_matches_input(self):
        """La salida debe tener exactamente la misma forma que la entrada."""
        from src.neural_manifold.unet_topology import UNet3D

        model = UNet3D(in_channels=1, out_channels=1)
        model.eval()

        # Tensor sintético: batch=1, canales=1, 64x64x64 vóxeles
        x = torch.randn(1, 1, 64, 64, 64)

        with torch.no_grad():
            y = model(x)

        assert y.shape == x.shape, (
            f"Ruptura de simetría dimensional: entrada {x.shape} → salida {y.shape}. "
            f"La UNet3D debe preservar la resolución espacial."
        )

    def test_output_range_sigmoid(self):
        """La salida debe estar en el rango [0, 1] (post-Sigmoid)."""
        from src.neural_manifold.unet_topology import UNet3D

        model = UNet3D(in_channels=1, out_channels=1)
        model.eval()

        x = torch.randn(1, 1, 64, 64, 64)

        with torch.no_grad():
            y = model(x)

        assert y.min() >= 0.0, f"Valor mínimo fuera de rango: {y.min():.4f} (esperado ≥ 0)"
        assert y.max() <= 1.0, f"Valor máximo fuera de rango: {y.max():.4f} (esperado ≤ 1)"

    def test_parameter_count(self):
        """Verifica que la red tiene el orden de magnitud esperado de parámetros (~1.4M)."""
        from src.neural_manifold.unet_topology import UNet3D

        model = UNet3D(in_channels=1, out_channels=1)
        total_params = sum(p.numel() for p in model.parameters())

        # Debe estar entre 500K y 5M (margen amplio para modificaciones menores)
        assert 500_000 < total_params < 5_000_000, (
            f"La red tiene {total_params:,} parámetros. "
            f"Esto está fuera del rango esperado [500K, 5M]."
        )


# ============================================================
# TEST 2: Consistencia de la Matriz Afín DICOM
# ============================================================
class TestAffineMatrix:
    """
    Verifica las propiedades algebraicas de la transformación afín T ∈ ℝ^{4×4}
    que mapea del espacio discreto del vóxel al espacio métrico euclidiano.
    Usa un dataset DICOM sintético para no depender de archivos reales.
    """

    def _create_synthetic_dicom(self, tmp_dir: str) -> str:
        """Crea un archivo DICOM mínimo válido para testing."""
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import pydicom.uid

        filepath = os.path.join(tmp_dir, "test_slice.dcm")

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.Rows = 512
        ds.Columns = 512
        ds.PixelSpacing = [0.75, 0.75]
        ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceThickness = 2.5
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((512, 512), dtype=np.int16).tobytes()
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']

        ds.save_as(filepath)
        return filepath

    def test_affine_is_4x4(self):
        """La matriz afín T debe ser exactamente 4×4."""
        from src.tensor_pde.io_module import extract_affine_matrix

        with tempfile.TemporaryDirectory() as tmp_dir:
            dcm_path = self._create_synthetic_dicom(tmp_dir)
            T = extract_affine_matrix(dcm_path)

            assert T.shape == (4, 4), f"Matriz afín tiene forma {T.shape}, esperada (4, 4)"

    def test_affine_last_row(self):
        """La última fila de T debe ser [0, 0, 0, 1] (transformación afín homogénea)."""
        from src.tensor_pde.io_module import extract_affine_matrix

        with tempfile.TemporaryDirectory() as tmp_dir:
            dcm_path = self._create_synthetic_dicom(tmp_dir)
            T = extract_affine_matrix(dcm_path)

            np.testing.assert_array_almost_equal(
                T[3, :], [0.0, 0.0, 0.0, 1.0],
                err_msg="La última fila de T no es [0,0,0,1]. La matriz no es afín homogénea."
            )

    def test_spacing_positive(self):
        """El spacing extraído de T debe ser estrictamente positivo en los 3 ejes."""
        from src.tensor_pde.io_module import extract_affine_matrix

        with tempfile.TemporaryDirectory() as tmp_dir:
            dcm_path = self._create_synthetic_dicom(tmp_dir)
            T = extract_affine_matrix(dcm_path)

            for axis in range(3):
                spacing = np.linalg.norm(T[axis, :3])
                assert spacing > 0, (
                    f"Spacing en eje {axis} es {spacing}. "
                    f"Un spacing ≤ 0 produciría mallas degeneradas."
                )


# ============================================================
# TEST 3: Sellado Topológico (Watertight)
# ============================================================
class TestWatertightSeal:
    """
    Verifica que el algoritmo de sellado topológico produce mallas cerradas
    (sin bordes abiertos), condición necesaria para que COMSOL pueda
    generar el mallado volumétrico tetraédrico.
    """

    def test_seal_produces_watertight_mesh(self):
        """Una esfera sintética con agujeros debe quedar sellada tras seal_geometry."""
        import trimesh
        from src.tensor_pde.mesh_repair import seal_geometry

        # Crear una esfera (ya es watertight) y abrirle un hueco
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)

        # Eliminar 20 caras para simular una malla abierta con agujeros
        faces_to_keep = sphere.faces[20:]
        broken_mesh = trimesh.Trimesh(vertices=sphere.vertices, faces=faces_to_keep)
        assert not broken_mesh.is_watertight, "La malla rota debería NO ser watertight (precondición del test)"

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "broken.stl")
            output_path = os.path.join(tmp_dir, "sealed.stl")

            broken_mesh.export(input_path)
            seal_geometry(input_path, output_path, pitch=1.0, smooth_iters=5)

            sealed = trimesh.load(output_path)
            assert sealed.is_watertight, (
                "La malla sellada NO es watertight. "
                "COMSOL rechazará esta geometría con errores de 'bordes abiertos'."
            )

    def test_seal_preserves_approximate_volume(self):
        """El volumen de la malla sellada debe mantener el orden de magnitud."""
        import trimesh
        from src.tensor_pde.mesh_repair import seal_geometry

        # Esfera grande con subdivisión alta para que la voxelización sea fiel
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=30.0)
        original_volume = abs(sphere.volume)

        # Daño mínimo: quitar solo 5 caras (agujero pequeño)
        faces_to_keep = sphere.faces[5:]
        broken_mesh = trimesh.Trimesh(vertices=sphere.vertices, faces=faces_to_keep)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "broken.stl")
            output_path = os.path.join(tmp_dir, "sealed.stl")

            broken_mesh.export(input_path)
            # pitch=2.0 igual que en producción (segment_pde.py)
            seal_geometry(input_path, output_path, pitch=2.0, smooth_iters=5)

            sealed = trimesh.load(output_path)
            sealed_volume = abs(sealed.volume)

            # Verificar que al menos se conserva el 10% del volumen original.
            # La re-voxelización siempre pierde algo de resolución, pero debe
            # mantener el orden de magnitud.
            ratio = sealed_volume / original_volume
            assert ratio > 0.10, (
                f"El volumen se redujo demasiado tras el sellado: "
                f"original={original_volume:.1f}, sellado={sealed_volume:.1f} (ratio={ratio:.2f})."
            )


# ============================================================
# TEST 4 (Bonus): DiceLoss diferenciable
# ============================================================
class TestDiceLoss:
    """
    Verifica las propiedades matemáticas de la función de pérdida Dice
    que guía el entrenamiento de la UNet3D.
    """

    def test_perfect_prediction_loss_zero(self):
        """Si la predicción es idéntica al ground truth, L_Dice debe ser ~0."""
        from src.neural_manifold.dataset_pde import DiceLoss

        loss_fn = DiceLoss()
        gt = torch.ones(1, 1, 16, 16, 16)
        pred = torch.ones(1, 1, 16, 16, 16)

        loss = loss_fn(pred, gt)
        assert loss.item() < 0.01, f"Pérdida con predicción perfecta: {loss.item():.4f} (esperado ~0)"

    def test_worst_prediction_loss_high(self):
        """Si la predicción es opuesta al ground truth, L_Dice debe ser ~1."""
        from src.neural_manifold.dataset_pde import DiceLoss

        loss_fn = DiceLoss()
        gt = torch.ones(1, 1, 16, 16, 16)
        pred = torch.zeros(1, 1, 16, 16, 16)

        loss = loss_fn(pred, gt)
        assert loss.item() > 0.9, f"Pérdida con predicción nula: {loss.item():.4f} (esperado ~1)"

    def test_loss_is_differentiable(self):
        """La pérdida debe ser diferenciable (backpropagation debe funcionar sin error)."""
        from src.neural_manifold.dataset_pde import DiceLoss

        loss_fn = DiceLoss()
        pred = torch.sigmoid(torch.randn(1, 1, 16, 16, 16, requires_grad=True))
        gt = torch.randint(0, 2, (1, 1, 16, 16, 16)).float()

        loss = loss_fn(pred, gt)
        loss.backward()  # Si esto no lanza error, la pérdida es diferenciable

if __name__ == "__main__":
    pytest.main([__file__, "-v"])