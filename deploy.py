import datetime
import importlib
import math
import os
import uuid

from google.cloud import aiplatform


common_util = importlib.import_module(
    "vertex-ai-samples.community-content.vertex_model_garden.model_oss.notebook_util.common_util"
)


# Get the default cloud project id.
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]

REGION = os.environ["GOOGLE_CLOUD_REGION"]

# Enable the Vertex AI API and Compute Engine API, if not already.
print("Enabling Vertex AI API and Compute Engine API.")
! gcloud services enable aiplatform.googleapis.com compute.googleapis.com

# Cloud Storage bucket for storing the experiment artifacts.
# A unique GCS bucket will be created for the purpose of this notebook. If you
# prefer using your own GCS bucket, change the value yourself below.
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
BUCKET_NAME = "/".join(BUCKET_URI.split("/")[:3])

if BUCKET_URI is None or BUCKET_URI.strip() == "" or BUCKET_URI == "gs://":
    BUCKET_URI = f"gs://{PROJECT_ID}-tmp-{now}-{str(uuid.uuid4())[:4]}"
    BUCKET_NAME = "/".join(BUCKET_URI.split("/")[:3])
    ! gsutil mb -l {REGION} {BUCKET_URI}
else:
    assert BUCKET_URI.startswith("gs://"), "BUCKET_URI must start with `gs://`."
    shell_output = ! gsutil ls -Lb {BUCKET_NAME} | grep "Location constraint:" | sed "s/Location constraint://"
    bucket_region = shell_output[0].strip().lower()
    if bucket_region != REGION:
        raise ValueError(
            "Bucket region %s is different from notebook region %s"
            % (bucket_region, REGION)
        )
print(f"Using this GCS Bucket: {BUCKET_URI}")

STAGING_BUCKET = os.path.join(BUCKET_URI, "temporal")
MODEL_BUCKET = os.path.join(BUCKET_URI, "stable_diffusion_v2_1")


# Initialize Vertex AI API.
print("Initializing Vertex AI API.")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

# Gets the default SERVICE_ACCOUNT.
shell_output = ! gcloud projects describe $PROJECT_ID
project_number = shell_output[-1].split(":")[1].strip().replace("'", "")
SERVICE_ACCOUNT = f"{project_number}-compute@developer.gserviceaccount.com"
print("Using this default Service Account:", SERVICE_ACCOUNT)

models, endpoints = {}, {}

# The pre-built serving docker image. It contains serving scripts and models.
TEXT_TO_IMAGE_DOCKER_URI = "us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/pytorch-inference.cu125.0-1.ubuntu2204.py310"


def deploy_model(model_id, task, accelerator_type, machine_type, accelerator_count=1):
    """Create a Vertex AI Endpoint and deploy the specified model to the endpoint."""
    common_util.check_quota(
        project_id=PROJECT_ID,
        region=REGION,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        is_for_training=False,
    )

    model_name = model_id
    endpoint = aiplatform.Endpoint.create(display_name=f"{model_name}-{task}-endpoint")
    serving_env = {
        "MODEL_ID": model_id,
        "TASK": task,
        "DEPLOY_SOURCE": "notebook",
    }

    if task == "image-to-image":
        model = aiplatform.Model.upload(
            display_name=model_name,
            serving_container_image_uri=IMAGE_TO_IMAGE_DOCKER_URI,
            serving_container_ports=[7080],
            serving_container_predict_route="/predictions/diffusers_serving",
            serving_container_health_route="/ping",
            serving_container_environment_variables=serving_env,
        )
    else:
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
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        deploy_request_timeout=1800,
        service_account=SERVICE_ACCOUNT,
    )
    print("To load this existing endpoint from a different session:")
    print(
        f'endpoint = aiplatform.Endpoint("projects/{PROJECT_ID}/locations/{REGION}/endpoints/{endpoint.name}")'
    )
    return model, endpoint



model_id = "stabilityai/stable-diffusion-2-1"

task = "text-to-image"  
accelerator_type = "NVIDIA_L4" 

if accelerator_type == "NVIDIA_L4":
    machine_type = "g2-standard-8"
elif accelerator_type == "NVIDIA_A100_80GB":
    machine_type = "a2-ultragpu-1g"
else:
    raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

common_util.check_quota(
    project_id=PROJECT_ID,
    region=REGION,
    accelerator_type=accelerator_type,
    accelerator_count=1,
    is_for_training=False,
)

models["image_model"], endpoints["image_model"] = deploy_model(
    model_id=model_id,
    task=task,
    accelerator_type=accelerator_type,
    machine_type=machine_type,
)
print("endpoint_name:", endpoints["image_model"].name)


if task == "text-to-image":
    comma_separated_prompt_list = "A futuristic cityscape at sunset"  # @param {type: "string"}
    prompt_list = [x.strip() for x in comma_separated_prompt_list.split(",")]
    negative_prompt = ""  # @param {type: "string"}
    height = 768  # @param {type:"number"}
    width = 768  # @param {type:"number"}
    num_inference_steps = 25  # @param {type:"number"}
    guidance_scale = 7.5  # @param {type:"number"}

    instances = [{"text": prompt} for prompt in prompt_list]
    parameters = {
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 7.5,
    }

    response = endpoints["image_model"].predict(
        instances=instances, parameters=parameters
    )
    images = [
        common_util.base64_to_image(prediction.get("output"))
        for prediction in response.predictions
    ]
    display(common_util.image_grid(images, rows=math.ceil(len(images) ** 0.5)))
else:
    print(
        "To run `text-to-image` prediction, deploy the model with `text-to-image` task."
    )
