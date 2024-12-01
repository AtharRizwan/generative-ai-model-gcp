from flask import Flask, request, jsonify, send_file
from torch import autocast
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Ensure output directory exists
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Stable Diffusion Text-to-Image API!"})

@app.route("/generate", methods=["POST"])
def generate_image():
    # Get text prompt from the request
    data = request.get_json()
    prompt = data.get("prompt")
    steps = data.get("steps", 50)
    guidance_scale = data.get("guidance_scale", 7.5)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Generate image
        with autocast("cuda"):
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale).images[0]

        # Save the image
        image_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
        image.save(image_path)

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
