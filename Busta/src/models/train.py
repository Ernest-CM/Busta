import json
from typing import Dict, Tuple

import numpy as np
from tensorflow import keras

from src.data.ingestion import discover_image_records, split_records
from src.evaluation.evaluate import evaluate_predictions
from src.models.cnn_model import build_baseline_cnn, set_global_seed
from src.preprocessing.preprocessor import Preprocessor
from src.utils.config import DataConfig, PreprocessConfig, TrainConfig
from src.utils.io_utils import ensure_directories


def _history_to_serializable(history: keras.callbacks.History) -> Dict[str, list]:
    return {key: [float(x) for x in values] for key, values in history.history.items()}


def train_baseline_model(
    data_config: DataConfig,
    preprocess_config: PreprocessConfig | None = None,
    train_config: TrainConfig | None = None,
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

    x_train, y_train, _ = preprocessor.preprocess_records(train_records)
    x_val, y_val, _ = preprocessor.preprocess_records(val_records)
    x_test, y_test, _ = preprocessor.preprocess_records(test_records)

    if x_train.size == 0 or x_val.size == 0 or x_test.size == 0:
        raise ValueError("Insufficient data after preprocessing; training/validation/test cannot be empty.")

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
        verbose=0,
    )

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

    return model, history_dict, evaluation.metrics


