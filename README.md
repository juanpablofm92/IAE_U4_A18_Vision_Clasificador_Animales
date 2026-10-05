# Actividad 18: Clasificador de Especies Animales (CNN & MobileNetV2)

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 📌 1. Descripción del Proyecto

Este proyecto aborda la tarea de clasificación supervisada de imágenes de fauna mediante visión artificial en `TensorFlow/Keras`. Se comparan dos paradigmas de aprendizaje profundo:
1. **Red Neuronal Convolucional (CNN) Personalizada:** 3 bloques convolucionales con `Conv2D`, `BatchNormalization`, `MaxPooling2D` y capas densas con `Dropout`.
2. **Transfer Learning (MobileNetV2):** Reutilización de representaciones visuales preentrenadas sobre el corpus ImageNet con congelamiento de capas y ajuste fino (*fine-tuning*).

El pipeline incluye aumento de datos (*Data Augmentation*):
* Rotación de 20 grados
* Zoom aleatorio de 0.2
* Volteo horizontal (*horizontal flip*)
* Desplazamiento horizontal y vertical del 10%

---

## 📂 2. Estructura del Proyecto

* `dataset_loader.py`: Generador y aumentador de imágenes con `tf.keras.preprocessing.image.ImageDataGenerator`. Incluye auto-generador de dataset de control si no existen carpetas locales.
* `train_eval.py`: Entrenamiento, evaluación con matriz de confusión Seaborn, reporte F1-score por especie y exportación a formato moderno `.keras`.
* `requirements.txt`: Dependencias del proyecto.

---

## 🚀 3. Instrucciones de Ejecución

```bash
pip install -r requirements.txt
python train_eval.py
```

---

## ⚖️ 4. Reflexión Ética en Visión Artificial

1. **Sesgo en Datos de Entrenamiento:** Los modelos entrenados con razas o especies predominantes pueden exhibir un rendimiento desproporcionadamente bajo en fauna endémica regional o silvestre en peligro.
2. **Impacto Ambiental de Grandes Modelos:** El uso de técnicas de *Transfer Learning* reduce sustancialmente el consumo energético y la huella de carbono asociada al entrenamiento de redes profundas desde cero.
