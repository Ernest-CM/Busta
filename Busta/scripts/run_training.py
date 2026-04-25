from pathlib import Path

from src.models.train import train_baseline_model
from src.utils.config import DataConfig, PreprocessConfig, TrainConfig

if __name__ == "__main__":
    dataset_path = Path("data") / "raw" / "cell_images"
    preprocess_config = PreprocessConfig(image_size=(224, 224), denoise_mode="median")
    data_config = DataConfig(dataset_root=dataset_path, test_size=0.2, random_state=42, stratify=True)
    train_config = TrainConfig(
        model_dir=Path("models_artifacts"),
        model_name="baseline_cnn.keras",
        history_name="training_history.json",
        epochs=5,
        batch_size=32,
        seed=42,
    )
    _, history, metrics = train_baseline_model(
        data_config=data_config,
        preprocess_config=preprocess_config,
        train_config=train_config,
    )
    print(f"Saved model: {(train_config.model_dir / train_config.model_name).resolve()}")
    print(f"Saved history: {(train_config.model_dir / train_config.history_name).resolve()}")
    print(f"Final train loss: {history['loss'][-1]:.6f}")
    print(f"Test metrics: {metrics}")

