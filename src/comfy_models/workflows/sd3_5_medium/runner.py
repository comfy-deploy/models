import uuid
from comfy_models.base.base_app import (
    _ComfyDeployRunnerModelsDownloadOptimzedImports,
)
import modal
from comfy_models.workflows import (
    get_configs,
)

from comfy_models.base.comfy_utils import Config

config = Config(
    id="sd3-5-medium",
    name="SD3.5 (Medium)",
    models_to_cache=[
        "checkpoints/sd3.5_medium.safetensors",
        "clip/clip_l.safetensors",
        "clip/clip_g.safetensors",
        "clip/t5xxl_fp16.safetensors",
    ],
    warmup_workflow=True,
    preview_image="https://comfy-deploy-output.s3.amazonaws.com/outputs/runs/36febfce-3cb6-4220-9447-33003e58d381/ComfyUI_00001_.png",
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
