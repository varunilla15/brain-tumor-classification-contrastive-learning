"""Self-supervised contrastive pretraining pipeline for brain CT classification.

Expected dataset layout:

data/
  train/class_a/*.png
  train/class_b/*.png
  val/class_a/*.png
  val/class_b/*.png
  test/class_a/*.png
  test/class_b/*.png

The class folder names are used as labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_dataset(split_dir: Path) -> tuple[list[str], list[int], list[str]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    class_names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in {split_dir}")

    image_paths: list[str] = []
    labels: list[int] = []

    for label, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(path))
                labels.append(label)

    if not image_paths:
        raise ValueError(f"No images found in {split_dir}")

    return image_paths, labels, class_names


def watershed_mask(gray: np.ndarray) -> np.ndarray:
    """Create a rough foreground/tumor-region mask using Otsu and Watershed."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_background = cv2.dilate(opening, kernel, iterations=3)

    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return binary

    _, sure_foreground = cv2.threshold(distance, 0.25 * distance.max(), 255, 0)
    sure_foreground = np.uint8(sure_foreground)
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_image, markers)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    if cv2.countNonZero(mask) < 25:
        return binary
    return mask


def preprocess_ct_image(
    image_path: str | Path,
    image_size: int,
    apply_segmentation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    image_path = str(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unable to read image: {image_path}")

    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    mask = watershed_mask(normalized)
    model_input = (
        cv2.bitwise_and(normalized, normalized, mask=mask)
        if apply_segmentation
        else normalized
    )

    rgb = cv2.cvtColor(model_input, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    return rgb, mask


def make_dataset(
    image_paths: list[str],
    labels: list[int],
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    paths_tensor = tf.constant(image_paths)
    labels_tensor = tf.constant(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            reshuffle_each_iteration=True,
        )

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image, mask_label = tf.py_function(
            func=lambda p, y: (
                preprocess_ct_image(p.numpy().decode("utf-8"), image_size)[0],
                y,
            ),
            inp=[path, label],
            Tout=[tf.float32, tf.int32],
        )
        image.set_shape((image_size, image_size, 3))
        mask_label.set_shape(())
        return image, mask_label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.GaussianNoise(0.03),
        ],
        name="contrastive_augmentation",
    )


def make_contrastive_dataset(
    supervised_dataset: tf.data.Dataset,
    augmentation: tf.keras.Model,
) -> tf.data.Dataset:
    def create_pair(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        del labels
        view_1 = augmentation(images, training=True)
        view_2 = augmentation(images, training=True)
        return view_1, view_2

    return supervised_dataset.map(create_pair, num_parallel_calls=tf.data.AUTOTUNE)


def build_encoder(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="ct_image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(256, activation="relu", name="features")(x)

    return tf.keras.Model(inputs, outputs, name="ct_cnn_encoder")


def build_projection_model(encoder: tf.keras.Model, image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="projection_input")
    x = encoder(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128)(x)
    outputs = tf.keras.layers.Lambda(
        lambda tensor: tf.math.l2_normalize(tensor, axis=1),
        name="l2_projection",
    )(x)
    return tf.keras.Model(inputs, outputs, name="contrastive_projection_model")


def nt_xent_loss(
    z_i: tf.Tensor,
    z_j: tf.Tensor,
    temperature: float = 0.1,
) -> tf.Tensor:
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)
    z = tf.math.l2_normalize(z, axis=1)

    similarity = tf.matmul(z, z, transpose_b=True) / temperature
    logits = similarity - (tf.eye(2 * batch_size) * 1e9)

    labels = tf.concat(
        [tf.range(batch_size, 2 * batch_size), tf.range(0, batch_size)],
        axis=0,
    )

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )
    return tf.reduce_mean(loss)


class ContrastiveTrainer(tf.keras.Model):
    def __init__(self, projection_model: tf.keras.Model, temperature: float) -> None:
        super().__init__()
        self.projection_model = projection_model
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.loss_tracker]

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        view_1, view_2 = data
        with tf.GradientTape() as tape:
            z_1 = self.projection_model(view_1, training=True)
            z_2 = self.projection_model(view_2, training=True)
            loss = nt_xent_loss(z_1, z_2, self.temperature)

        gradients = tape.gradient(loss, self.projection_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.projection_model.trainable_variables)
        )
        self.loss_tracker.update_state(loss)
        return {"contrastive_loss": self.loss_tracker.result()}


def build_classifier(
    encoder: tf.keras.Model,
    image_size: int,
    num_classes: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="ct_image")
    x = encoder(inputs)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="brain_tumor_classifier")


def plot_training_curves(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_frame.get("accuracy", []), label="train")
    axes[0].plot(history_frame.get("val_accuracy", []), label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history_frame.get("loss", []), label="train")
    axes[1].plot(history_frame.get("val_loss", []), label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> None:
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_prediction_artifacts(
    model: tf.keras.Model,
    image_path: Path,
    class_names: list[str],
    image_size: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image, mask = preprocess_ct_image(image_path, image_size)
    probabilities = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_AREA)
    original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    mask_color = np.zeros_like(original_color)
    mask_color[:, :, 2] = mask
    overlay = cv2.addWeighted(original_color, 0.75, mask_color, 0.35, 0)

    cv2.imwrite(str(output_dir / "segmentation_mask.png"), mask)
    cv2.imwrite(str(output_dir / "tumor_overlay.png"), overlay)

    predicted_class = class_names[predicted_index]
    result = {
        "image": str(image_path),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(class_names)
        },
    }
    (output_dir / "prediction.json").write_text(json.dumps(result, indent=2))

    print(f"Prediction: {predicted_class} ({confidence:.4f})")
    print(f"Artifacts saved to: {output_dir}")


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_paths, train_labels, class_names = discover_dataset(data_dir / "train")
    val_paths, val_labels, val_class_names = discover_dataset(data_dir / "val")

    if class_names != val_class_names:
        raise ValueError("Train and validation class folders must match.")

    num_classes = len(class_names)
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    train_dataset = make_dataset(
        train_paths,
        train_labels,
        args.image_size,
        args.batch_size,
        training=True,
    )
    val_dataset = make_dataset(
        val_paths,
        val_labels,
        args.image_size,
        args.batch_size,
        training=False,
    )

    encoder = build_encoder(args.image_size)

    if not args.skip_pretrain:
        augmentation = build_augmentation()
        contrastive_dataset = make_contrastive_dataset(train_dataset, augmentation)
        projection_model = build_projection_model(encoder, args.image_size)
        contrastive_trainer = ContrastiveTrainer(projection_model, args.temperature)
        contrastive_trainer.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate)
        )
        contrastive_trainer.fit(
            contrastive_dataset,
            epochs=args.epochs_pretrain,
            verbose=1,
        )

    classifier = build_classifier(encoder, args.image_size, num_classes)
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "brain_tumor_classifier.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    history = classifier.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs_finetune,
        callbacks=callbacks,
        verbose=1,
    )

    classifier.save(output_dir / "brain_tumor_classifier.keras")
    plot_training_curves(history, output_dir)

    test_dir = data_dir / "test"
    if test_dir.exists():
        test_paths, test_labels, test_class_names = discover_dataset(test_dir)
        if test_class_names != class_names:
            raise ValueError("Train and test class folders must match.")
        test_dataset = make_dataset(
            test_paths,
            test_labels,
            args.image_size,
            args.batch_size,
            training=False,
        )
        evaluate_model(classifier, test_dataset, class_names, output_dir)

    print(f"Training complete. Artifacts saved to: {output_dir}")


def predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model) if args.model else Path(args.output_dir) / "brain_tumor_classifier.keras"
    class_names_path = Path(args.class_names) if args.class_names else Path(args.output_dir) / "class_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(class_names_path.read_text())
    save_prediction_artifacts(
        model=model,
        image_path=Path(args.predict),
        class_names=class_names,
        image_size=args.image_size,
        output_dir=Path(args.output_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain CT classification with contrastive pretraining."
    )
    parser.add_argument("--data-dir", default="data", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="outputs", help="Artifact directory.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs-pretrain", type=int, default=20)
    parser.add_argument("--epochs-finetune", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", help="Path to a single image for prediction.")
    parser.add_argument("--model", help="Path to a trained .keras model.")
    parser.add_argument("--class-names", help="Path to class_names.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
