import json
import os
import importlib
from typing import Dict, Any, List, Literal, Optional, Union

import modal
from pydantic import BaseModel

from comfy_models.base.comfy_utils import Config
from comfy_models.base.comfy_utils import generate_modal_image
from comfy_models.base.inputs import get_inputs_from_workflow_api
from comfy_models.base.outputs import get_outputs_from_workflow

COMFY_DEPENDENCIES = [
    "BennyKok/ComfyUI@2697a11",
    "ty0x2333/ComfyUI-Dev-Utils@0dac07c",
    "BennyKok/comfyui-deploy@a82e315",
]

def get_volumes(safe_model_name: str):
    MODAL_VOLUME_NAME = os.getenv("MODAL_VOLUME_NAME")
    volumes = {  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/root/.nv": modal.Volume.from_name(
            "nv-cache-" + safe_model_name, create_if_missing=True
        ),
        "/root/.triton": modal.Volume.from_name(
            "triton-cache-" + safe_model_name, create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache-" + safe_model_name, create_if_missing=True
        ),
    }
    if MODAL_VOLUME_NAME is not None:
        volumes["/private_models"] = modal.Volume.from_name(
            MODAL_VOLUME_NAME, create_if_missing=True
        )
    return volumes


def get_configs(config: Config):
    return {
        "image": generate_modal_image(dependencies=COMFY_DEPENDENCIES),
        "container_idle_timeout": 300,
        "timeout": 60 * 60,  # leave plenty of time for compilation
        "secrets": [modal.Secret.from_name("hf-models-download")],
        "volumes": get_volumes(config.id),
        "enable_memory_snapshot": True,
        "mounts": [
            modal.Mount.from_local_file(
                local_path="workflow_api.json",
                remote_path="/root/workflow/workflow_api.json",
            )
        ],
        "gpu": config.gpu,
    }


class ModelInput(BaseModel):
    input_id: str
    class_type: Union[
        str,
        Literal[
            "ComfyUIDeployExternalText",
            "ComfyUIDeployExternalTextAny",
            "ComfyUIDeployExternalTextSingleLine",
            "ComfyUIDeployExternalImage",
            "ComfyUIDeployExternalImageAlpha",
            "ComfyUIDeployExternalNumber",
            "ComfyUIDeployExternalNumberInt",
            "ComfyUIDeployExternalLora",
            "ComfyUIDeployExternalCheckpoint",
            "ComfyDeployWebscoketImageInput",
            "ComfyUIDeployExternalImageBatch",
            "ComfyUIDeployExternalVideo",
            "ComfyUIDeployExternalBoolean",
            "ComfyUIDeployExternalNumberSlider",
            "ComfyUIDeployExternalNumberSliderInt",
            "ComfyUIDeployExternalEnum",
        ],
    ]
    required: bool
    default_value: Optional[Any] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    display_name: Optional[str] = None
    description: Optional[str] = None


class ModelOutput(BaseModel):
    class_type: Literal["ComfyDeployStdOutputImage", "ComfyDeployStdOutputAny"]
    output_id: str


class WorkflowConfig(Config):
    inputs: List[ModelInput] = []
    outputs: List[ModelOutput] = []


def get_all_workflow_configs() -> Dict[str, WorkflowConfig]:
    """
    Dynamically imports and returns all runner.py files from workflow subdirectories
    """
    workflows_dir = os.path.dirname(__file__)
    workflow_configs = {}

    # Get all subdirectories in the workflows directory
    for item in sorted(os.listdir(workflows_dir)):
        workflow_path = os.path.join(workflows_dir, item)

        # Skip if not a directory or if it's __pycache__ or starts with _
        if not os.path.isdir(workflow_path) or item.startswith("_"):
            continue

        # Check if runner.py exists in the workflow directory
        config_path = os.path.join(workflow_path, "runner.py")
        workflow_api_path = os.path.join(workflow_path, "workflow_api.json")
        if not os.path.exists(config_path):
            continue

        try:
            # Import the config module
            module_path = f"comfy_models.workflows.{item}.runner"
            config_module = importlib.import_module(module_path)
            # print(module_path)

            # Get the config object
            if hasattr(config_module, "config"):
                # Convert Config object to dict before unpacking
                base_config = config_module.config.dict()
                workflow_configs[item] = base_config

                # Load and process workflow API if it exists
            if os.path.exists(workflow_api_path):
                with open(workflow_api_path, "r") as f:
                    workflow_api = f.read()

                # Get inputs and outputs
                inputs = get_inputs_from_workflow_api(workflow_api)
                outputs = get_outputs_from_workflow(workflow_api)

                workflow_configs[item] = WorkflowConfig(
                    **workflow_configs[item],
                    inputs=[ModelInput(**input, required=True) for input in inputs],
                    outputs=[ModelOutput(**output) for output in outputs],
                )

        except Exception as e:
            print(f"Error loading config from {item}: {str(e)}")

    return workflow_configs
