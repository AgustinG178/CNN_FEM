# Pipeline Biomecánico: DICOM a Elementos Finitos (FEM) con IA
Este repositorio contiene la arquitectura de software desarrollada para automatizar la reconstrucción tridimensional y el análisis biomecánico de estructuras óseas. Mediante Redes Neuronales Convolucionales 3D (UNet3D), se extrae la topología ósea de tomografías computarizadas (CT/DICOM) y se exporta como mallas isótropas optimizadas para COMSOL Multiphysics.

> [!IMPORTANT]
> **Hito de Fase 2:** El sistema ha superado la divergencia de coordenadas LPS/RAS. El entrenamiento actual demuestra una convergencia acelerada (Dice Score >58% en Época 8) gracias a la sincronización estricta de ejes.

---

## 🗂️ Estructura del Proyecto
El código se organiza en módulos especializados por responsabilidad física y computacional:

*   **`src/neural_manifold/`**: Motores de IA. Arquitectura UNet3D, lógica de ventana deslizante e inferencia topológica mediante **parches 3D** (sub-volúmenes de $64^3$ vóxeles que permiten procesar tomografías de alta resolución por partes).
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

---

## 💾 Gestión de Datos y Peso del Repositorio

Si clonas este repositorio, notarás que la carpeta `data/` está vacía o ausente. Esto es **intencional**.

Debido a que el set de datos médicos completo (CTs crudos + parches de entrenamiento) supera los **40 GB**, estos archivos están protegidos por `.gitignore` y no se suben a la nube. Para reconstruir el entorno de datos, debes:
1. Colocar tus DICOMs en `data/01_raw/`.
2. Ejecutar `python prepare_dataset.py`.
3. Esto generará localmente la estructura de parches necesaria para el entrenamiento.

---

## 🏛️ Despliegue en Clúster de Alto Rendimiento (HPC)

Este pipeline está diseñado para ejecutarse en nodos de supercómputo que utilizan gestores de colas **SLURM**.
1. **Configuración:** El archivo `run_cluster.slurm` contiene las directivas para solicitar recursos (CPUs, Memoria, Tiempo).
2. **Lanzamiento:** `sbatch run_cluster.slurm`
3. **Monitoreo:** `tail -f entrenamiento_[JOB_ID].log`
