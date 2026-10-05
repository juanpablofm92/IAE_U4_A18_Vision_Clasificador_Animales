"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 18: Visión Artificial - Clasificador de Animales (Entrenamiento y Transfer Learning)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dataset_loader import obtener_generadores

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
except ImportError:
    tf = None


def construir_cnn_personalizada(input_shape=(128, 128, 3), num_classes=3) -> tf.keras.Model:
    """Arquitectura CNN profunda personalizada con regularización por Dropout."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def construir_modelo_transfer_learning(input_shape=(128, 128, 3), num_classes=3) -> tf.keras.Model:
    """Modelo preentrenado con MobileNetV2 usando Feature Extraction y Fine-Tuning."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congelar capas base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluar_y_reportar(model, test_generator, class_labels, titulo="Modelo"):
    print(f"\n" + "=" * 60)
    print(f"  EVALUACIÓN DE DESEMPEÑO: {titulo}")
    print("=" * 60)

    test_generator.reset()
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    print("\n[+] Reporte de Clasificación (F1-score por especie):")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Matriz de Confusión - {titulo}")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.tight_layout()
    output_img = f"matriz_confusion_{titulo.lower().replace(' ', '_')}.png"
    plt.savefig(output_img, dpi=200)
    plt.close()
    print(f"[+] Gráfica de matriz de confusión guardada como: '{output_img}'")


def main():
    if tf is None:
        print("[!] TensorFlow no se encuentra instalado en el entorno. Por favor ejecute: pip install tensorflow")
        return

    print("=" * 70)
    print("  TECNM / ITSU - CLASIFICADOR DE ESPECIES ANIMALES")
    print("=" * 70)

    train_gen, test_gen = obtener_generadores(img_size=(128, 128), batch_size=8)
    class_labels = list(train_gen.class_indices.keys())
    num_classes = len(class_labels)
    print(f"[*] Clases identificadas: {class_labels}")

    # 1. CNN Personalizada
    print("\n--- 1. Entrenamiento de CNN Personalizada ---")
    cnn_model = construir_cnn_personalizada(input_shape=(128, 128, 3), num_classes=num_classes)
    cnn_model.fit(train_gen, epochs=3, validation_data=test_gen, verbose=1)
    evaluar_y_reportar(cnn_model, test_gen, class_labels, "CNN Personalizada")

    # Exportación
    cnn_model.save("modelo_cnn_animales.keras")
    print("[+] Modelo CNN guardado en: 'modelo_cnn_animales.keras'")

    # 2. Transfer Learning (MobileNetV2)
    print("\n--- 2. Transfer Learning con MobileNetV2 ---")
    tl_model = construir_modelo_transfer_learning(input_shape=(128, 128, 3), num_classes=num_classes)
    tl_model.fit(train_gen, epochs=3, validation_data=test_gen, verbose=1)
    evaluar_y_reportar(tl_model, test_gen, class_labels, "Transfer Learning MobileNetV2")

    # Exportación
    tl_model.save("modelo_mobilenet_animales.keras")
    print("[+] Modelo MobileNetV2 guardado en: 'modelo_mobilenet_animales.keras'")


if __name__ == "__main__":
    main()
