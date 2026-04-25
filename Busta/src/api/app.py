from flask import Flask

from src.api.routes import api_bp


def create_app(model=None, preprocessor=None) -> Flask:
    app = Flask(__name__)
    app.config["MODEL"] = model
    app.config["PREPROCESSOR"] = preprocessor
    app.register_blueprint(api_bp, url_prefix="/api")
    return app

