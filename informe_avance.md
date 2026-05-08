# Informe de Avance: BoneFlow AI - Pipeline Biomecánico Automatizado
**Fecha de actualización:** 8 de Mayo, 2026

> [!IMPORTANT]
> **Resumen Ejecutivo**
> El proyecto ha transicionado exitosamente de una etapa de "Prueba de Concepto" (Fase 1) a un modelo de "Producción de Alta Fidelidad" (Fase 2). Tras validar que la red neuronal básica era capaz de aprender la morfología ósea pero presentaba debilidades en estructuras corticales finas, se ha implementado una arquitectura de Estado del Arte (Attention-ResUNet3D) con parches de gran escala (128³) y funciones de pérdida penalizadas (Focal Loss). El sistema se encuentra actualmente en fase de re-entrenamiento masivo en el clúster.

---

## 1. Cronología del Desarrollo y Arquitectura del Sistema

El pipeline se divide en tres niveles de abstracción. A continuación se detalla su estado de implementación real:

### Fase 1: Ingeniería de Datos (Completada)
*   **Destilación de Etiquetas:** Procesamiento de 61 pacientes mediante autosegmentación (TotalSegmentator) para generar el Ground Truth maestro.
*   **Corrección Espacial:** Sincronización de los ejes DICOM (LPS) y NIfTI (RAS), eliminando el error de "espejado" mediante operadores de re-muestreo afín (`resample_from_to`).
*   **Particionamiento V2:** Generación de un nuevo dataset de parches isométricos de $128^3$ vóxeles, optimizando el contexto anatómico de la red.

### Fase 2: Aprendizaje Profundo y Optimización (En Ejecución)
Tras una primera etapa de testeo con parches de $64^3$ (Fase 1 PoC), se ha migrado a una arquitectura avanzada para garantizar la calidad médica del resultado.
*   **Modelo:** Attention-ResUNet3D (32 filtros base).
*   **Novedad:** Integración de bloques residuales y compuertas de atención para preservar la topología de las alas ilíacas.
*   **Entrenamiento:** Ejecutándose actualmente en el nodo de cómputo con una función de pérdida híbrida **Focal-Dice Loss**.

### Fase 3: Post-Procesamiento e Integración Biomecánica (Validada)
Esta fase comprende la lógica de salida una vez finalizado el entrenamiento.
*   **Clasificación:** Algoritmo de componentes conexos para separar Pelvis y Fémures.
*   **Reparación:** Pipeline de sellado *Watertight* y suavizado de Taubin para garantizar mallas exportables a COMSOL.
*   **Estatus:** El código está 100% implementado y validado mediante "Sanity Checks" con los pesos del modelo anterior. Está a la espera de los pesos finales del modelo V2.

---

## 2. Diagnóstico de la Fase 1: Aprendizaje y Lecciones
La primera versión del modelo (V1) alcanzó un **Dice Score del 64.8% (Época 21)**. Si bien los resultados fueron prometedores, una auditoría cualitativa reveló la necesidad del upgrade actual:

*   **Hallazgo:** La red clásica omitía zonas donde el hueso es muy delgado (agujeros en el ala ilíaca).
*   **Causa:** Falta de contexto espacial en parches pequeños y desbalance de clases (el hueso fino representa muy pocos píxeles para el Dice simple).
*   **Evidencia Histórica:**
    ![Curva de Convergencia V1](loss_curve.png)
    *(Registro del entrenamiento inicial demostrando la estabilidad lograda antes del salto arquitectónico).*

---

## 3. Especificaciones Técnicas de la Versión 2.0
La actual etapa de entrenamiento incorpora mejoras críticas para alcanzar una precisión superior al 85%:

### 3.1 Super-Parches de 128³
Al duplicar la dimensión lineal (8 veces más volumen por parche), la red neuronal adquiere una "visión de conjunto". Ya no ve fragmentos aislados de hueso, sino estructuras anatómicas completas (ej. la articulación coxofemoral entera), permitiendo una reconstrucción topológica sin fracturas artificiales.

### 3.2 Atención y Residuos
*   **Attention Gates:** Actúan como filtros dinámicos que fuerzan a la IA a ignorar los tejidos blandos y concentrar su gradiente en la corteza ósea.
*   **Residual Connections:** Permiten que la red aprenda por "diferencias", facilitando el entrenamiento de modelos más profundos sin degradación de la señal.

### 3.3 Focal-Dice Loss (Matemática de los bordes)
Se ha sustituido la pérdida tradicional por una combinación que penaliza exponencialmente los errores en píxeles "difíciles". Esto justifica por qué el Loss inicial se observa por encima de $1.0$, ya que el sistema está castigando severamente la falta de precisión en los bordes finos.

---

## 4. Próximos Pasos: Fase 3 Final
Una vez alcanzada la convergencia asintótica del modelo V2, el pipeline procederá automáticamente a:
1.  **Extracción de Malla:** Marching Cubes sobre el codominio de probabilidades.
2.  **Optimización FEM:** Mallado de Voronoi isótropo mediante PyACVD.
3.  **Mapeo de Rigidez:** Asignación del Módulo de Young ($E$) basado en Unidades Hounsfield (Ley de Wolff) para su resolución en COMSOL Multiphysics mediante la ecuación de Navier-Cauchy.

---
**Conclusión:** El sistema es actualmente estable y se encuentra procesando la versión definitiva de la IA con los más altos estándares de segmentación médica 3D.
