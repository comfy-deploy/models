import uuid
from comfy_models.base.base_app import (
    _ComfyDeployRunnerModelsDownloadOptimzedImports,
)
import modal
from comfy_models.workflows import (
    get_configs,
)
from comfy_models.workflows.flux_dev.config import config

APP_NAME = config.name
app = modal.App(APP_NAME)


@app.cls(
    **get_configs(APP_NAME),
    enable_memory_snapshot=True,
)
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
