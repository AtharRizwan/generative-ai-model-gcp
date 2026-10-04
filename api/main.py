"""Cloud Run proxy between the CanvasCraft frontend and the Vertex AI endpoint.

Firebase Hosting rewrites /api/** to this service, so the browser never holds
Google credentials; this service calls Vertex AI with its own service account.
"""
import os

from flask import Flask, request, jsonify
from google.cloud import aiplatform

PROJECT_ID = os.environ["PROJECT_ID"]
ENDPOINT_ID = os.environ["ENDPOINT_ID"]
REGION = os.environ.get("REGION", "us-central1")
MAX_PROMPT_LENGTH = 500

app = Flask(__name__)

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "Prompt is required"}), 400
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({"error": f"Prompt must be at most {MAX_PROMPT_LENGTH} characters"}), 400

    try:
        response = endpoint.predict(
            instances=[{"text": prompt.strip()}],
            parameters={
                "negative_prompt": "",
                "height": 768,
                "width": 768,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
            },
        )
        return jsonify({"image": response.predictions[0]["output"]})
    except Exception:
        # Details go to Cloud Logging, not to the public client
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Image generation failed, please try again"}), 502
