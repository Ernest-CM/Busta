from pathlib import Path

from src.api.app import create_app
from src.models.predict import load_model_for_inference
from src.preprocessing.preprocessor import Preprocessor
from src.utils.config import PreprocessConfig

MODEL_PATH = Path("models_artifacts") / "baseline_cnn.keras"


def main() -> None:
    model = load_model_for_inference(MODEL_PATH)
    preprocessor = Preprocessor(config=PreprocessConfig())
    app = create_app(model=model, preprocessor=preprocessor)
    print("Malaria Screening API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
