import uuid
from comfy_models.base.base_app import (
    _ComfyDeployRunnerModelsDownloadOptimzedImports,
)
import modal
from comfy_models.base.comfy_utils import Config
from comfy_models.workflows import get_configs

config = Config(
    id="flux-schnell",
    name="Flux (Schnell)",
    # memroy snspshot
    models_to_cache=[
        "unet/flux1-schnell.sft",
        "clip/clip_l.safetensors",
        "clip/clip_g.safetensors",
        "clip/t5xxl_fp16.safetensors",
        "vae/ae.sft",
    ],
    warmup_workflow=True,
    preview_image="https://comfy-deploy-output-dev.s3.us-east-2.amazonaws.com/outputs/runs/b5afa7eb-a15f-4c45-a95c-d5ce89cb537f/image.jpeg",
)


app = modal.App(config.id)


@app.cls(**get_configs(config))
class ComfyDeployRunner(_ComfyDeployRunnerModelsDownloadOptimzedImports):
    config = config


@app.local_entrypoint()
def main():
    ComfyDeployRunner().run.remote(
        {
            "prompt_id": str(uuid.uuid4()),
            "inputs": {
                "positive_prompt": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
            },
        }
    )
