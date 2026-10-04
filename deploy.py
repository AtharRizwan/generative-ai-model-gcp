"""Deploy a Stable Diffusion model to a Vertex AI endpoint and run a test prediction.

Examples:
    # Deploy the fine-tuned model from Cloud Storage
    python deploy.py --model-id gs://BUCKET/models/sd21-pixelart

    # Only run a test prediction against an existing endpoint
    python deploy.py --endpoint-id 1234567890 --prompt "pixel art castle"
"""
import argparse
import base64
import io
import os
import subprocess

from google.cloud import aiplatform
from PIL import Image


# The pre-built serving docker image. It contains serving scripts and models.
TEXT_TO_IMAGE_DOCKER_URI = "us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/pytorch-inference.cu125.0-1.ubuntu2204.py310"

MACHINE_TYPES = {
    "NVIDIA_L4": "g2-standard-8",
    "NVIDIA_A100_80GB": "a2-ultragpu-1g",
}


def default_service_account(project_id):
    """Return the project's default Compute Engine service account."""
    project_number = subprocess.run(
        ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{project_number}-compute@developer.gserviceaccount.com"


def deploy_model(model_id, accelerator_type, service_account, accelerator_count=1):
    """Create a Vertex AI Endpoint and deploy the specified model to the endpoint."""
    task = "text-to-image"
    model_name = model_id.rstrip("/").split("/")[-1]
    endpoint = aiplatform.Endpoint.create(display_name=f"{model_name}-{task}-endpoint")
    serving_env = {
        "MODEL_ID": model_id,
        "TASK": task,
        "DEPLOY_SOURCE": "notebook",
    }

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=TEXT_TO_IMAGE_DOCKER_URI,
        serving_container_ports=[7080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables=serving_env,
    )

    model.deploy(
        endpoint=endpoint,
        machine_type=MACHINE_TYPES[accelerator_type],
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        deploy_request_timeout=1800,
        service_account=service_account,
    )
    return model, endpoint


def predict(endpoint, prompt, output_path):
    """Generate one image from the endpoint and save it to output_path."""
    instances = [{"text": prompt}]
    parameters = {
        "negative_prompt": "",
        "height": 768,
        "width": 768,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
    }
    response = endpoint.predict(instances=instances, parameters=parameters)
    image_bytes = base64.b64decode(response.predictions[0]["output"])
    Image.open(io.BytesIO(image_bytes)).save(output_path)
    print(f"Saved test image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1",
                        help="Hugging Face model ID or gs:// path to a diffusers model folder")
    parser.add_argument("--accelerator", choices=sorted(MACHINE_TYPES), default="NVIDIA_L4")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--region", default=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"))
    parser.add_argument("--endpoint-id", help="Skip deployment and test this existing endpoint")
    parser.add_argument("--prompt", default="A futuristic cityscape at sunset")
    parser.add_argument("--output", default="sample.png", help="Where to save the test image")
    args = parser.parse_args()

    if not args.project:
        parser.error("--project or GOOGLE_CLOUD_PROJECT is required")

    aiplatform.init(project=args.project, location=args.region)

    if args.endpoint_id:
        endpoint = aiplatform.Endpoint(
            f"projects/{args.project}/locations/{args.region}/endpoints/{args.endpoint_id}"
        )
    else:
        # Enable the Vertex AI API and Compute Engine API, if not already.
        print("Enabling Vertex AI API and Compute Engine API.")
        subprocess.run(
            ["gcloud", "services", "enable", "aiplatform.googleapis.com", "compute.googleapis.com",
             f"--project={args.project}"],
            check=True,
        )
        service_account = default_service_account(args.project)
        print("Using this default Service Account:", service_account)
        print(f"Deploying {args.model_id} on {args.accelerator}; this can take 15-30 minutes.")
        _, endpoint = deploy_model(args.model_id, args.accelerator, service_account)

    print("ENDPOINT_ID:", endpoint.name)
    predict(endpoint, args.prompt, args.output)


if __name__ == "__main__":
    main()
