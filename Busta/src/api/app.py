import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.api.limiter import limiter
from src.api.routes import api_bp
from src.api.ui import ui_bp

logger = logging.getLogger(__name__)


def create_app(model=None, preprocessor=None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MODEL"] = model
    app.config["PREPROCESSOR"] = preprocessor
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    limiter.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(ui_bp, url_prefix="")

    @app.before_request
    def _log_request():
        logger.info("Request: %s %s from %s", request.method, request.path, request.remote_addr)

    @app.after_request
    def _log_response(response):
        logger.info("Response: %s %d %s bytes", request.path, response.status_code, response.content_length)
        return response

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"error": "File too large. Maximum upload size is 16 MB."}), 413

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return jsonify({"error": "rate limit exceeded"}), 429

    return app
