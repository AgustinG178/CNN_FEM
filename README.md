# Pipeline Biomecánico: DICOM a Elementos Finitos (FEM) con IA

Este repositorio contiene la arquitectura de software desarrollada para automatizar la reconstrucción tridimensional y el análisis biomecánico de estructuras óseas (pelvis y fémur). Mediante Inteligencia Artificial (Redes Neuronales Convolucionales 3D), se extrae la topología ósea de tomografías computarizadas (CT/DICOM) y se exporta como mallas listas para simulaciones de Elementos Finitos en COMSOL Multiphysics.

> [!NOTE]
> Para una explicación exhaustiva sobre la matemática, las Ecuaciones en Derivadas Parciales de Navier-Cauchy y el modelo de convergencia de la Inteligencia Artificial, por favor lee el documento científico [informe_avance.md](./informe_avance.md).

---

## 🗂️ Estructura del Proyecto

El código está modularizado separando estrictamente la preparación de datos, la Inteligencia Artificial y la física del continuo.

```text
Automatizacion FEM/
├── data/                       # ⚠️ IGNORADA EN GITHUB (Ver nota abajo)
│   ├── 01_raw_dicom/           # Tomografías originales
│   ├── 03_models/              # Pesos pre-entrenados de la red (.pth)
│   └── 04_training_patches/    # Tensores 3D extraídos y filtrados
├── src/                        # Código fuente modular
│   ├── neural_manifold/        # Módulo IA: Arquitectura UNet3D, Auto-Labeler, Dataset y Loss
│   └── tensor_pde/             # Módulo Física: Mapeo de materiales, Meshing y COMSOL
├── logs/                       # Registros (Logs) de salida del clúster HPC
├── requirements_cluster.txt    # Dependencias exactas (Optimizadas para Python 3.6 en HPC)
├── run_cluster.slurm           # Orquestador de trabajos para SLURM Workload Manager
├── prepare_dataset.py          # Script principal para limpieza y generación de parches
├── informe_avance.md           # Informe de estado, justificación empírica y matemáticas
└── README.md                   # Esta guía
```

---

## 💾 Nota sobre los Datos (Por qué no está la carpeta `data/`)

Si descargas o clonas este repositorio, notarás que la carpeta `data/` y todos los archivos de extensión `.npy`, `.dcm` o `.pth` no están presentes. 

Esto es **intencional**. El set de datos tomográficos original y los parches extraídos tras el proceso de *Negative Sampling* superan los **30 GB** (llegando a 180 GB en su estado crudo), lo cual excede por mucho los límites arquitectónicos de GitHub. 

**Para reproducir el entrenamiento:**
1. Deberás colocar tus propios archivos médicos en `data/01_raw_dicom/`.
2. Ejecutar localmente el script `python prepare_dataset.py`.
3. Esto destilará el conocimiento mediante *TotalSegmentator*, aislará las regiones de interés y generará automáticamente la estructura pesada de carpetas que la IA necesita.

---

## 🚀 Instalación y Despliegue en Clúster HPC

Este pipeline está fuertemente optimizado para ser ejecutado en nodos de supercómputo que utilizan gestores de colas **SLURM**, permitiendo entrenamiento distribuido en CPU utilizando OpenMP.

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd TU_REPOSITORIO
   ```

2. **Instalar Dependencias:**
   Asegúrate de utilizar las versiones exactas provistas para garantizar la compatibilidad de PyTorch y Torchio con intérpretes Python legados en servidores.
   ```bash
   python3 -m pip install --user -r requirements_cluster.txt
   ```

3. **Lanzar el entrenamiento:**
   ```bash
   sbatch run_cluster.slurm
   ```
   Puedes monitorear el progreso y la caída de la función de pérdida matemática (*Dice Loss*) utilizando:
   ```bash
   tail -f logs/entrenamiento_*.log
   ```
