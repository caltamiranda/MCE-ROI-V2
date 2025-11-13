import os
import random

def select_10_percent(input_path, output_path, percent=0.1):
    """
    Lee un archivo de labels, toma un porcentaje aleatorio,
    y lo guarda en un nuevo archivo.
    """

    # Leer labels
    with open(input_path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    # Seleccionar 10%
    sample_size = max(1, int(len(labels) * percent))
    selected = random.sample(labels, sample_size)

    # Guardar archivo
    with open(output_path, "w") as f:
        f.write("\n".join(selected))

    print(f"[OK] {os.path.basename(output_path)} generado ({sample_size} labels)")


def main():
    base_dir = "splits"   # donde están tus labels originales

    files = [
        ("labels_train.txt", "labels_train_10.txt"),
        ("labels_val.txt",   "labels_val_10.txt"),
        ("labels_test.txt",  "labels_test_10.txt"),
    ]

    for infile, outfile in files:
        input_path = os.path.join(base_dir, infile)
        output_path = os.path.join(base_dir, outfile)

        if not os.path.exists(input_path):
            print(f"[WARN] No encontrado: {input_path}")
            continue

        select_10_percent(input_path, output_path, percent=0.10)


if __name__ == "__main__":
    import random
    random.seed(42)  # reproducible
    main()
