"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 18: Visión Artificial - Clasificador de Especies de Animales
Módulo: Preparación de Datos y Aumento de Datos (Data Augmentation)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Importación condicional de TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    tf = None
    ImageDataGenerator = None

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_animales")
CLASES_DEFECTO = ("perro", "gato", "caballo")


def crear_dataset_animales_si_no_existe(base_dir: str = DEFAULT_DATA_DIR,
                                        clases: Tuple[str, ...] = CLASES_DEFECTO,
                                        muestras_por_clase: int = 25) -> None:
    """
    Tarea 1: Recolectar y organizar dataset de imágenes de animales estructurado
    en particiones 'train' y 'test'. Si no existe un conjunto físico local,
    genera imágenes sintéticas calibradas para pruebas automatizadas.
    """
    if os.path.exists(base_dir) and os.path.isdir(os.path.join(base_dir, "train")):
        return

    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    colores = {
        "perro": (160, 82, 45),     # Café siena
        "gato": (255, 140, 0),      # Naranja atardecer
        "caballo": (70, 130, 180)   # Azul acero
    }

    n_train = int(muestras_por_clase * 0.8)
    n_test = muestras_por_clase - n_train

    for split, count in [("train", n_train), ("test", n_test)]:
        for clase in clases:
            dir_clase = os.path.join(base_dir, split, clase)
            os.makedirs(dir_clase, exist_ok=True)
            for i in range(count):
                img = Image.new("RGB", (128, 128), color=(245, 245, 245))
                draw = ImageDraw.Draw(img)
                base_color = colores.get(clase, (100, 100, 100))
                ruido = np.random.randint(-15, 16, size=3)
                color = tuple(np.clip(np.array(base_color) + ruido, 0, 255))

                if clase == "perro":
                    # Silueta ovalada con orejas caídas
                    draw.ellipse([25 + (i % 5), 25, 103 - (i % 5), 95], fill=color)
                    draw.polygon([(25, 30), (15, 60), (35, 50)], fill=color)
                    draw.polygon([(100, 30), (110, 60), (90, 50)], fill=color)
                elif clase == "gato":
                    # Silueta con orejas puntiagudas triangulares
                    draw.ellipse([30, 35, 98, 95], fill=color)
                    draw.polygon([(35, 40), (25, 15), (55, 35)], fill=color)
                    draw.polygon([(93, 40), (103, 15), (73, 35)], fill=color)
                else:  # caballo
                    # Silueta alargada de crin y cabeza
                    draw.polygon([(40, 20), (88, 20), (98, 85), (30, 85)], fill=color)
                    draw.rectangle([35, 80, 55, 115], fill=color)

                img.save(os.path.join(dir_clase, f"{clase}_{i:03d}.png"))

    print(f"[+] Dataset estructurado de animales listo en: '{base_dir}'")


def generar_mosaico_aumento_datos(ruta_imagen_muestra: str = None,
                                  ruta_salida: str = "mosaico_data_augmentation.png") -> str:
    """
    Tarea 1: Visualizar el pipeline de Data Augmentation aplicando transformaciones
    geométricas y fotométricas a una imagen de referencia.
    """
    if ruta_imagen_muestra is None or not os.path.exists(ruta_imagen_muestra):
        # Tomar una imagen generada
        crear_dataset_animales_si_no_existe()
        ruta_imagen_muestra = os.path.join(DEFAULT_DATA_DIR, "train", "gato", "gato_000.png")

    img_pil = Image.open(ruta_imagen_muestra).convert("RGB")
    
    transformaciones = [
        ("Original", img_pil),
        ("Rotación +20°", img_pil.rotate(20, expand=False, fillcolor=(245, 245, 245))),
        ("Rotación -20°", img_pil.rotate(-20, expand=False, fillcolor=(245, 245, 245))),
        ("Flip Horizontal", img_pil.transpose(Image.FLIP_LEFT_RIGHT)),
        ("Zoom (Crop 80%)", img_pil.crop((12, 12, 116, 116)).resize((128, 128))),
        ("Brillo +30%", Image.eval(img_pil, lambda x: min(255, int(x * 1.3))))
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle("Tarea 1: Pipeline de Data Augmentation (Técnicas Aplicadas)", fontsize=14, fontweight="bold")
    
    for ax, (titulo, im) in zip(axes.flatten(), transformaciones):
        ax.imshow(im)
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=180)
    plt.close()
    print(f"[+] Mosaico de Data Augmentation exportado: '{ruta_salida}'")
    return ruta_salida


def obtener_generadores(data_dir: str = DEFAULT_DATA_DIR,
                        img_size: Tuple[int, int] = (128, 128),
                        batch_size: int = 8):
    """
    Tarea 1: Construir generadores de datos para entrenamiento y validación
    con normalización y aumento de datos en tiempo de ejecución.
    """
    crear_dataset_animales_si_no_existe(data_dir)

    if ImageDataGenerator is not None:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode="nearest"
        )
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        train_path = os.path.join(data_dir, "train")
        test_path = os.path.join(data_dir, "test")

        train_gen = train_datagen.flow_from_directory(
            train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )
        test_gen = test_datagen.flow_from_directory(
            test_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )
        return train_gen, test_gen
    else:
        # Modo generador NumPy liviano
        return cargar_dataset_numpy(data_dir, img_size)


def cargar_dataset_numpy(data_dir: str, img_size: Tuple[int, int] = (128, 128)):
    """Carga de tensores en NumPy para entornos sin TensorFlow."""
    X_train, y_train = [], []
    X_test, y_test = [], []
    clases = sorted([d for d in os.listdir(os.path.join(data_dir, "train"))
                     if os.path.isdir(os.path.join(data_dir, "train", d))])
    class_indices = {c: i for i, c in enumerate(clases)}

    for split, X_list, y_list in [("train", X_train, y_train), ("test", X_test, y_test)]:
        split_dir = os.path.join(data_dir, split)
        for c in clases:
            cdir = os.path.join(split_dir, c)
            for fname in os.listdir(cdir):
                if fname.endswith(".png"):
                    p = os.path.join(cdir, fname)
                    img = Image.open(p).convert("RGB").resize(img_size)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    X_list.append(arr)
                    y_list.append(class_indices[c])

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test)), class_indices


if __name__ == "__main__":
    print("[*] Ejecutando módulo de preparación y aumento de datos...")
    crear_dataset_animales_si_no_existe()
    generar_mosaico_aumento_datos()
    print("[+] Tarea 1 verificada exitosamente.")
