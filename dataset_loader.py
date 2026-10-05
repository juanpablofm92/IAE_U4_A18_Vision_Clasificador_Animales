"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 18: Visión Artificial - Clasificador de Animales (Data Loader & Augmentation)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import os
import numpy as np
from typing import Tuple

# Intentar importar TensorFlow, con fallback seguro si no está instalado
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    tf = None
    ImageDataGenerator = None


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_animales")

def crear_dataset_sintetico_si_no_existe(base_dir: str = DEFAULT_DATA_DIR,
                                         clases=("perro", "gato", "caballo"),
                                         muestras_por_clase: int = 15):
    """
    Crea un conjunto estructurado de imágenes PNG sintéticas con patrones geométricos
    diferenciados para garantizar que el pipeline se ejecute sin dependencias externas complejas.
    """
    if os.path.exists(base_dir):
        return

    from PIL import Image, ImageDraw

    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    colores = {
        "perro": (160, 82, 45),     # Café
        "gato": (255, 140, 0),      # Naranja
        "caballo": (70, 130, 180)   # Azul acero
    }

    for subsplit, num_imgs in [("train", int(muestras_por_clase * 0.8)), ("test", int(muestras_por_clase * 0.2))]:
        for idx_c, clase in enumerate(clases):
            dir_clase = os.path.join(base_dir, subsplit, clase)
            os.makedirs(dir_clase, exist_ok=True)
            for i in range(num_imgs):
                img = Image.new("RGB", (128, 128), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)
                # Formas características
                base_color = colores[clase]
                ruido = np.random.randint(-20, 20)
                color = tuple(np.clip(np.array(base_color) + ruido, 0, 255))
                
                if clase == "perro":
                    draw.ellipse([20 + i, 20, 100 - i, 100], fill=color)
                elif clase == "gato":
                    draw.polygon([(64, 20 + i), (20, 100), (108, 100)], fill=color)
                else: # caballo
                    draw.rectangle([25, 25 + i, 95, 95 - i], fill=color)

                img.save(os.path.join(dir_clase, f"{clase}_{i:03d}.png"))
    
    print(f"[+] Dataset sintético de control creado en: '{base_dir}'")


def obtener_generadores(data_dir: str = DEFAULT_DATA_DIR,
                        img_size: Tuple[int, int] = (128, 128),
                        batch_size: int = 8):
    """
    Construye generadores de datos con Data Augmentation:
    - Rotación de 20 grados
    - Zoom de rango 0.2
    - Volteo horizontal aleatorio
    - Normalización [0, 1]
    """
    crear_dataset_sintetico_si_no_existe(data_dir)

    if ImageDataGenerator is None:
        raise ImportError("TensorFlow/Keras no está disponible en este entorno.")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator


if __name__ == "__main__":
    print("[*] Verificando cargador de datos y Data Augmentation...")
    crear_dataset_sintetico_si_no_existe()
    print("[+] Generador preparado correctamente.")
