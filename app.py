from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch
import os
import uuid

# Initialize Flask app
app = Flask(__name__)

# Load the Stable Diffusion model (set MODEL_ID to a local path such as
# "fine-tuned-stable-diffusion" to serve the fine-tuned model)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model_id = os.environ.get("MODEL_ID", "CompVis/stable-diffusion-v1-4")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)
# Lower peak memory so the model fits on small (~6 GB) GPUs
pipe.enable_attention_slicing()

# Ensure output directory exists
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Stable Diffusion Text-to-Image API!"})

@app.route("/generate", methods=["POST"])
def generate_image():
    # Get text prompt from the request
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not prompt or not isinstance(prompt, str):
        return jsonify({"error": "Prompt is required"}), 400

    try:
        steps = int(data.get("steps", 50))
        guidance_scale = float(data.get("guidance_scale", 7.5))
    except (TypeError, ValueError):
        return jsonify({"error": "steps must be an integer and guidance_scale a number"}), 400

    if not 1 <= steps <= 100:
        return jsonify({"error": "steps must be between 1 and 100"}), 400

    try:
        # Generate image
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale).images[0]

        # Save the image under a random name; the prompt is user input and
        # must not be used as a path
        image_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.png")
        image.save(image_path)

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
