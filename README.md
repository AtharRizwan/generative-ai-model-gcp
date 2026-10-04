# generative-ai-model-gcp

**CanvasCraft** is a web app that turns text into pixel-art images. Stable Diffusion 2.1 is fine-tuned with LoRA on a pixel-art dataset, served from a Vertex AI endpoint, and called from a Firebase-hosted page through a small Cloud Run proxy.

```
upload_dataset.ipynb  →  gs://pixel-art-dataset/dataset.zip
fine_tune.ipynb       →  gs://pixel-art-dataset/models/sd21-pixelart   (LoRA merged into SD 2.1)
deploy.py             →  Vertex AI endpoint (NVIDIA L4)
api/                  →  Cloud Run service that calls the endpoint
frontend/             →  Firebase Hosting page; /api/** is routed to the Cloud Run service
```

The browser only talks to its own site, so no Google credentials ever reach it.

## Repository layout

| Path | What it is |
| --- | --- |
| `upload_dataset.ipynb` | Downloads 1,000 samples of [`jainr3/diffusiondb-pixelart`](https://huggingface.co/datasets/jainr3/diffusiondb-pixelart) and uploads them to Cloud Storage |
| `fine_tune.ipynb` | Trains LoRA adapters on the SD 2.1 UNet, merges them into a full model and uploads it |
| `deploy.py` | Deploys a model to a Vertex AI endpoint and saves a test image |
| `api/` | Flask proxy for Cloud Run: `POST /api/generate` |
| `frontend/` | Static Firebase Hosting site |
| `app.py` | Optional local Flask server that runs Stable Diffusion in-process (not used by the website) |

## Prerequisites

- A Google Cloud project with billing enabled and quota for one NVIDIA L4 GPU in `us-central1`
- The [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) and the [Firebase CLI](https://firebase.google.com/docs/cli)
- Python 3.10–3.12
- A CUDA GPU for fine-tuning, such as an L4 (24 GB) on Colab or Vertex AI Workbench

Install the Python dependencies. Install torch from the PyTorch index that matches your CUDA version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The commands below use `pixel-art-dataset` as the bucket and `us-central1` as the region. If you use your own bucket, change its name in both notebooks.

## 1. Prepare the dataset

Run `upload_dataset.ipynb`. It saves the images and a `labels.csv` (columns `text,image_path`) to `dataset/`, zips the folder and uploads it to `gs://pixel-art-dataset/dataset.zip`.

The notebooks access Cloud Storage with your Application Default Credentials. Set them up as follows:
- **On your own machine:** run `gcloud auth application-default login`.
- **On Colab:** run `from google.colab import auth; auth.authenticate_user()` first.
- **On Vertex AI Workbench:** nothing to do; credentials are already available.

## 2. Fine-tune

Run `fine_tune.ipynb` on a GPU machine. All the settings are in its **Configuration** cell:

| Setting | Default | Notes |
| --- | --- | --- |
| `resolution` | `768` | Matches the deployed endpoint. Drop to `512` if you run out of GPU memory. |
| `batch_size`, `gradient_accumulation_steps` | `1`, `4` | Effective batch size of 4 |
| `learning_rate`, `num_epochs`, `lora_rank` | `1e-4`, `1`, `8` | |
| `model_gcs_uri` | `gs://pixel-art-dataset/models/sd21-pixelart` | Where the merged model is uploaded |

The notebook does the following:
1. Trains LoRA adapters.
2. Saves them to `sd21-pixelart-lora/`.
3. Merges them into a full model in `fine-tuned-stable-diffusion/`.
4. Writes `comparison.png`, with base-model output on the left and fine-tuned output on the right.
5. Uploads the merged model to `model_gcs_uri`.

## 3. Deploy the model to Vertex AI

The Vertex AI serving container runs as the project's default compute service account, so that account needs read access to the bucket:

```bash
export GOOGLE_CLOUD_PROJECT=<your-project-id>
PROJECT_NUMBER=$(gcloud projects describe $GOOGLE_CLOUD_PROJECT --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding gs://pixel-art-dataset \
  --member=serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/storage.objectViewer

python deploy.py --model-id gs://pixel-art-dataset/models/sd21-pixelart
```

Deployment takes 15–30 minutes. When it finishes, the script prints the `ENDPOINT_ID` and saves a test image to `sample.png`.

| Option | Default |
| --- | --- |
| `--model-id` | `stabilityai/stable-diffusion-2-1` (a Hugging Face ID or a `gs://` path) |
| `--accelerator` | `NVIDIA_L4` (or `NVIDIA_A100_80GB`) |
| `--project` | `$GOOGLE_CLOUD_PROJECT` (required) |
| `--region` | `$GOOGLE_CLOUD_REGION`, or `us-central1` if unset |
| `--endpoint-id` | Skip deployment and only test an existing endpoint |
| `--prompt`, `--output` | Test prompt and image path |

To test an endpoint without redeploying:

```bash
python deploy.py --endpoint-id <ENDPOINT_ID> --prompt "pixel art castle on a hill"
```

If the deployment can't load the model from Cloud Storage, push the merged model to the Hugging Face Hub instead with `pipeline.push_to_hub("<user>/sd21-pixelart")`. Then pass that repo ID as `--model-id`.

## 4. Deploy the API proxy to Cloud Run

Firebase Hosting can only route to Cloud Run services in the same project, so deploy into your Firebase project. The service gets its own service account, which is only allowed to call Vertex AI:

```bash
gcloud iam service-accounts create canvascraft-api
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member=serviceAccount:canvascraft-api@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com \
  --role=roles/aiplatform.user

gcloud run deploy canvascraft-api --source api --region us-central1 \
  --service-account canvascraft-api@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com \
  --allow-unauthenticated --max-instances 2 \
  --set-env-vars PROJECT_ID=$GOOGLE_CLOUD_PROJECT,ENDPOINT_ID=<ENDPOINT_ID>
```

| Environment variable | Required | Default |
| --- | --- | --- |
| `PROJECT_ID` | yes | |
| `ENDPOINT_ID` | yes | |
| `REGION` | no | `us-central1` |

**API:** `POST /api/generate` with `{"prompt": "..."}` (at most 500 characters) returns `{"image": "<base64-encoded image>"}`. On failure it returns `{"error": "..."}` with status 400 for a bad prompt or 502 if generation fails.

To run the proxy locally against a deployed endpoint:

```bash
gcloud auth application-default login
cd api && pip install -r requirements.txt
PROJECT_ID=<project> ENDPOINT_ID=<ENDPOINT_ID> flask --app main run
curl -X POST localhost:5000/api/generate -H 'Content-Type: application/json' -d '{"prompt": "pixel art cat"}'
```

## 5. Deploy the website

`frontend/.firebaserc` points at the Firebase project `cloud-asg2`. Change it if you're deploying to your own project.

```bash
cd frontend
firebase hosting:channel:deploy preview   # temporary preview URL; /api/** routing works here too
firebase deploy --only hosting            # publish to the live site
```

## Running Stable Diffusion locally (optional)

`app.py` serves a model from your own machine on port 5000. It uses the GPU if one is available and the CPU otherwise.

```bash
python app.py                                          # CompVis/stable-diffusion-v1-4
MODEL_ID=fine-tuned-stable-diffusion python app.py     # the merged model from step 2
FLASK_DEBUG=1 python app.py                            # Flask debug mode (never on a public port)

curl -X POST localhost:5000/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "pixel art cat", "steps": 25, "guidance_scale": 7.5}' -o out.png
```

`steps` must be between 1 and 100. Images are also saved to `generated_images/`.

## Costs and cleanup

A deployed endpoint bills for its GPU node every hour it stays deployed, even with no traffic. To remove it:

```bash
gcloud ai endpoints list --region=us-central1
DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe <ENDPOINT_ID> --region=us-central1 --format="value(deployedModels[0].id)")
gcloud ai endpoints undeploy-model <ENDPOINT_ID> --region=us-central1 --deployed-model-id=$DEPLOYED_MODEL_ID
gcloud ai endpoints delete <ENDPOINT_ID> --region=us-central1
```

Cloud Run scales to zero when idle. `--max-instances 2` limits how much traffic the public site can send to the endpoint.
