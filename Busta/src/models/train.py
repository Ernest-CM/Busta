import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.ingestion import discover_image_records, split_records
from src.evaluation.evaluate import evaluate_predictions
from src.models.cnn_model import build_baseline_cnn, set_global_seed
from src.models.transfer_model import build_transfer_model, unfreeze_top_layers
from src.preprocessing.augmentation import build_augmentation_pipeline
from src.preprocessing.preprocessor import Preprocessor
from src.utils.config import AUGMENTATION_SEED, FINE_TUNE_LAYERS, DataConfig, PreprocessConfig, TrainConfig
from src.utils.io_utils import ensure_directories


def _history_to_serializable(history: keras.callbacks.History) -> Dict[str, list]:
    return {key: [float(x) for x in values] for key, values in history.history.items()}


def train_baseline_model(
    data_config: DataConfig,
    preprocess_config: PreprocessConfig | None = None,
    train_config: TrainConfig | None = None,
    processed_dir: Path | None = None,
) -> Tuple[keras.Model, Dict[str, list], Dict[str, float | None]]:
    """
    Train baseline CNN on preprocessed NIH-style dataset and persist artifacts.

    Returns:
    - trained model
    - history dictionary
    - test evaluation metrics dictionary
    """
    preprocess_config = preprocess_config or PreprocessConfig()
    train_config = train_config or TrainConfig()
    ensure_directories([train_config.model_dir])

    set_global_seed(train_config.seed)
    preprocessor = Preprocessor(config=preprocess_config)

    all_records = discover_image_records(dataset_root=data_config.dataset_root)
    train_records, test_records = split_records(
        records=all_records,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
        stratify=data_config.stratify,
    )
    train_records, val_records = split_records(
        records=train_records,
        test_size=train_config.val_size,
        random_state=train_config.seed,
        stratify=True,
    )

    print(f"[1/5] Preprocessing {len(train_records)} training images...")
    x_train, y_train, _ = preprocessor.preprocess_records(train_records)
    print(f"[2/5] Preprocessing {len(val_records)} validation images...")
    x_val, y_val, _ = preprocessor.preprocess_records(val_records)
    print(f"[3/5] Preprocessing {len(test_records)} test images...")
    x_test, y_test, _ = preprocessor.preprocess_records(test_records)

    if x_train.size == 0 or x_val.size == 0 or x_test.size == 0:
        raise ValueError("Insufficient data after preprocessing; training/validation/test cannot be empty.")

    print(f"[4/5] Building model and starting training ({train_config.epochs} epochs)...")
    model = build_baseline_cnn(
        input_shape=(preprocess_config.image_size[0], preprocess_config.image_size[1], 3),
        learning_rate=train_config.learning_rate,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=train_config.batch_size,
        epochs=train_config.epochs,
        verbose=1,
    )
    print("[5/5] Evaluating and saving artifacts...")

    history_dict = _history_to_serializable(history)
    model_path = train_config.model_dir / train_config.model_name
    history_path = train_config.model_dir / train_config.history_name
    eval_path = train_config.model_dir / train_config.evaluation_name
    cm_path = train_config.model_dir / train_config.confusion_matrix_name

    model.save(model_path)
    history_path.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")

    y_score = model.predict(x_test, verbose=0).reshape(-1)
    y_pred = (y_score >= 0.5).astype(np.int64)
    evaluation = evaluate_predictions(y_true=y_test, y_pred=y_pred, y_score=y_score)
    eval_path.write_text(json.dumps(evaluation.metrics, indent=2), encoding="utf-8")
    np.savetxt(cm_path, evaluation.confusion_matrix, delimiter=",", fmt="%d")

    if processed_dir is not None:
        ensure_directories([processed_dir])
        np.save(processed_dir / "X_test.npy", x_test)
        np.save(processed_dir / "y_test.npy", y_test)

    return model, history_dict, evaluation.metrics


def _merge_histories(h1: Dict[str, list], h2: Dict[str, list]) -> Dict[str, list]:
    merged: Dict[str, List[float]] = {}
    for key in h1:
        merged[key] = h1[key] + h2.get(key, [])
    return merged


def train_transfer_model(
    data_dir,
    output_dir,
    base: str = "mobilenetv2",
    epochs: int = 10,
    fine_tune_epochs: int = 5,
    batch_size: int = 32,
) -> Tuple[keras.Model, Dict[str, list]]:
    """
    Two-stage transfer learning pipeline.

    Stage 1: frozen base, train head only with augmentation.
    Stage 2: unfreeze top layers, fine-tune with lower learning rate.

    Args:
        data_dir: root directory of raw images (Parasitized/ and Uninfected/ subdirs).
        output_dir: directory to save model and history JSON.
        base: base architecture — 'mobilenetv2' or 'efficientnetb0'.
        epochs: max epochs for stage 1.
        fine_tune_epochs: max epochs for stage 2.
        batch_size: batch size for both stages.

    Returns:
        (trained model, merged history dict)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    ensure_directories([output_dir])

    preprocess_config = PreprocessConfig()
    preprocessor = Preprocessor(config=preprocess_config)

    all_records = discover_image_records(dataset_root=data_dir)
    train_records, test_records = split_records(all_records, test_size=0.2, random_state=42, stratify=True)
    train_records, val_records = split_records(train_records, test_size=0.2, random_state=42, stratify=True)

    x_train, y_train, _ = preprocessor.preprocess_records(train_records)
    x_val, y_val, _ = preprocessor.preprocess_records(val_records)

    if x_train.size == 0 or x_val.size == 0:
        raise ValueError("Insufficient data — training or validation set is empty.")

    aug = build_augmentation_pipeline(seed=AUGMENTATION_SEED)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=42)
        .batch(batch_size)
        .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    input_shape = (preprocess_config.image_size[0], preprocess_config.image_size[1], 3)
    model = build_transfer_model(input_shape=input_shape, base=base, freeze_base=True)

    # Stage 1 — train head with frozen base
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )

    # Stage 2 — fine-tune top layers of base
    unfreeze_top_layers(model, n_layers=FINE_TUNE_LAYERS)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )

    history = _merge_histories(
        {k: [float(v) for v in vals] for k, vals in h1.history.items()},
        {k: [float(v) for v in vals] for k, vals in h2.history.items()},
    )

    model_path = output_dir / f"transfer_{base}.keras"
    history_path = output_dir / f"transfer_{base}_history.json"
    model.save(str(model_path))
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return model, history
