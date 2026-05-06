# Pipeline Biomecánico: DICOM a Elementos Finitos (FEM) con IA
Este repositorio contiene la arquitectura de software desarrollada para automatizar la reconstrucción tridimensional y el análisis biomecánico de estructuras óseas. Mediante Redes Neuronales Convolucionales 3D (UNet3D), se extrae la topología ósea de tomografías computarizadas (CT/DICOM) y se exporta como mallas isótropas optimizadas para COMSOL Multiphysics.

> [!IMPORTANT]
> **Hito de Fase 2:** El sistema ha superado la divergencia de coordenadas LPS/RAS. El entrenamiento actual demuestra una convergencia acelerada (Dice Score >58% en Época 8) gracias a la sincronización estricta de ejes.

---

## 🗂️ Estructura del Proyecto
El código se organiza en módulos especializados por responsabilidad física y computacional:

*   **`src/neural_manifold/`**: Motores de IA. Arquitectura UNet3D, lógica de ventana deslizante e inferencia topológica.
*   **`src/tensor_pde/`**: Motores de Física. Mapeo HU → Young, reparación de mallas (Watertight) y optimización de calidad mediante partición de Voronoi.
*   **`scripts/`**: Utilidades de validación rápida y generación de visuales para informes.

---

## 🚀 Guía de Uso Rápido (Local)

### 1. Preparación del Dataset (Fase 1)
Si tienes tomografías nuevas, colócalas en `data/01_raw/` y corre:
```bash
python prepare_dataset.py
```
Esto generará las máscaras de Ground Truth y los parches de entrenamiento filtrados por densidad.

### 2. Prueba de Inferencia (Validación de Pesos)
Para probar un modelo entrenado (ej. de la Época 8) sobre un paciente:
1. Descarga el `.pth` del clúster a `data/03_models/`.
2. Ejecuta el test de inferencia:
```bash
python scripts/test_inference.py
```
*Este script aplica automáticamente un filtro de densidad (>200 HU) y genera una malla STL limpia.*

### 3. Visualización de Alineación
Para generar capturas 2D que verifiquen la sincronización entre DICOM y Máscara:
```bash
python scripts/create_report_visuals.py
```

---

## 🏗️ Tecnología y Optimización FEM
A diferencia de segmentadores genéricos, este pipeline está diseñado para **Ingeniería Biomecánica**:
*   **Remallado de Voronoi:** Integra `pyacvd` para garantizar que los elementos de la malla sean isótropos, evitando errores de convergencia en el mallador de COMSOL.
*   **Sellado Watertight:** Algoritmo determinista que asegura que la geometría sea un sólido cerrado (2-variedad), eliminando auto-intersecciones.
*   **Mapeo de Materiales:** Genera campos escalares de Módulo de Young basados en la Ley de Wolff, permitiendo simulaciones de heterogeneidad ósea real.

---

## ⚙️ Instalación
Asegúrate de tener un entorno Python 3.10+ y ejecuta:
```bash
pip install -r requirements.txt
```
*Nota: Para la optimización de mallas se requieren `pyacvd` y `pyvista`, ya incluidos en las dependencias.*

## 🏛️ Despliegue en Clúster (SLURM)
Para entrenar el modelo en un entorno de alto rendimiento:
```bash
sbatch run_cluster.slurm
```
Monitoreo de convergencia: `tail -f entrenamiento_[JOB_ID].log`
