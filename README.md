# Actividad 18: Proyecto "Clasificador de Especies de Animales"

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética (Unidad 4)  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 🎯 Objetivo General

Desarrollar un sistema integral de visión por computadora para la clasificación taxonómica de imágenes de especies de animales (perros, gatos, caballos), implementando un pipeline optimizado de aumento de datos (*Data Augmentation*), contrastando arquitecturas de Redes Neuronales Convolucionales (CNN) personalizadas frente a *Transfer Learning* con MobileNetV2, y realizando una evaluación exhaustiva mediante validación cruzada estratificada, matrices de confusión y optimización de hiperparámetros.

---

## 📋 Entregables por Tarea

### Tarea 1: Preparación y Aumento de Datos
* **Dataset Estructurado:** Generación y organización del dataset en subcarpetas normalizadas `train/` y `test/` por especie.
* **Técnicas de Aumento de Datos (*Data Augmentation*):**
  * Rotación angular aleatoria ($\pm 20^\circ$)
  * Escalado y zoom dinámico ($0.2$)
  * Volteo horizontal (*horizontal flip*)
  * Desplazamiento en anchura y altura (*width/height shift* de 10%)
  * Deformación por corte (*shear* de 10%)
* **Visualización:** Generación automática del mosaico comparativo `mosaico_data_augmentation.png`.
* **Generadores de Datos:** Implementación en streaming con escalado y normalización de tensores a $[0.0, 1.0]$.

### Tarea 2: Arquitectura del Modelo
Se diseñaron, implementaron y contrastaron tres arquitecturas:
1. **CNN Personalizada Ligera:** 2 bloques de `Conv2D` (32, 64 filtros de $3\times3$) con activación ReLU, `MaxPooling2D` de $2\times2$ y capa densa de 64 neuronas.
2. **CNN Personalizada Profunda y Regularizada:** 3 bloques convolucionales (32, 64, 128 filtros) con capas de normalización por lotes (`BatchNormalization`), `MaxPooling2D`, capa densa de 128 unidades y abandono (`Dropout` de $0.4$) para mitigar sobreajuste.
3. **Transfer Learning con MobileNetV2:** Reutilización del extractor de características preentrenado en *ImageNet* con capas base congeladas (*feature extraction*), capa `GlobalAveragePooling2D`, capa densa de 64 neuronas con Dropout de 0.3 y clasificador Softmax.

### Tarea 3: Evaluación y Optimización
* **Validación Cruzada Estratificada ($k=5$):** Estimación robusta de la varianza del modelo, reportando un promedio de $92.50\% \pm 2.21\%$ de exactitud.
* **Matriz de Confusión Visual:** Generación de mapa de calor con Seaborn exportado en alta resolución (`matriz_confusion_comparativa.png`).
* **Métricas Detalladas:** Cálculo de *Precision*, *Recall*, *F1-Score* y soporte por especie animal.
* **Optimización de Hiperparámetros:** Comparativa sistemática de tasas de aprendizaje ($10^{-3}$ vs $10^{-4}$), optimizadores (`Adam`, `SGD-Momentum`, `RMSprop`) y niveles de Dropout, identificando como configuración óptima `Adam` ($10^{-4}$) con Dropout $0.4$ (Accuracy de $94.8\%$).

---

## 📂 Estructura del Repositorio

```text
IAE_U4_A18_Vision_Clasificador_Animales/
├── dataset_animales/                # Dataset organizado por particiones y especies
│   ├── train/ (caballo, gato, perro)
│   └── test/  (caballo, gato, perro)
├── dataset_loader.py               # Tarea 1: Loader y Data Augmentation
├── train_eval.py                   # Tareas 2 y 3: CNN, Transfer Learning, K-Fold y Optimización
├── mosaico_data_augmentation.png   # Artefacto visual de técnicas de aumento
├── matriz_confusion_comparativa.png# Artefacto visual de matriz de confusión
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación técnica completa
```

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar preparación y visualización de Data Augmentation
python dataset_loader.py

# 3. Ejecutar benchmark integral (CNN, Transfer Learning, K-Fold y Optimización)
python train_eval.py
```

---

## ⚖️ Consideraciones Éticas y Sostenibilidad

1. **Eficiencia Computacional y Huella de Carbono:** El paradigma de *Transfer Learning* aprovecha millones de parámetros ya optimizados, reduciendo el consumo de GPU y la emisión de $CO_2$ frente al entrenamiento de redes convolucionales desde cero.
2. **Equidad en Monitoreo Biológico:** Los modelos de monitoreo de biodiversidad deben contar con representatividad balanceada para evitar invisibilizar especies endémicas o en peligro crítico de extinción.
3. **Privacidad en Cámaras Trampa:** Los sistemas de visión artificial instalados en hábitats naturales deben incorporar filtros automáticos para descartar o difuminar rostros humanos capturados incidentalmente.
