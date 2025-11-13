import random
import os

def generate_splits(
        total=60000,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        output_dir="."
    ):
    """
    Genera listas aleatorias de labels en formato 00000–59999
    y las divide en train/val/test con los porcentajes indicados.
    """

    # ---- 1. Generar labels tipo 00000, 00001, ..., 59999 ----
    labels = [f"{i:05d}" for i in range(total)]

    # ---- 2. Mezclar aleatoriamente ----
    random.shuffle(labels)

    # ---- 3. Calcular tamaños ----
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_labels = labels[:n_train]
    val_labels = labels[n_train:n_train + n_val]
    test_labels = labels[n_train + n_val:]

    # ---- 4. Crear carpeta si no existe ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- 5. Guardar archivos ----
    with open(os.path.join(output_dir, "labels_train.txt"), "w") as f:
        f.write("\n".join(train_labels))

    with open(os.path.join(output_dir, "labels_val.txt"), "w") as f:
        f.write("\n".join(val_labels))

    with open(os.path.join(output_dir, "labels_test.txt"), "w") as f:
        f.write("\n".join(test_labels))

    print(f"[OK] Generado en: {output_dir}")
    print(f" Train: {len(train_labels)}")
    print(f" Val:   {len(val_labels)}")
    print(f" Test:  {len(test_labels)}")


if __name__ == "__main__":
    generate_splits(output_dir="splits")
