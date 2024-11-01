# Models API by ComfyDeploy

A collection of optimized ComfyUI-based cloud inference endpoints, built on [ComfyDeploy](https://github.com/BennyKok/comfyui-deploy) and [Modal](https://modal.com/).

## Features

- Optimized model loading and caching
- Cloud-based inference endpoints
- Easy workflow management through CLI
- Extensible workflow system

## Installation

```bash
uv add comfy-models
```

## Usage

### CLI Commands

```bash
uv tool install comfy-models
```

The package provides several CLI commands to manage workflows:

```bash
# List all available workflows
models ls

# Show detailed info about a workflow
models cat <workflow_name>

# Deploy a workflow to Modal
models deploy <workflow_name>

# Run a workflow locally
models run <workflow_name>

# Test a workflow
models test <workflow_name>
```

### Environment Setup

Create a `.env` file in your project root:

```env
MODAL_VOLUME_NAME=your_modal_volume_name
HF_TOKEN=your_huggingface_token  # Optional, for private model access
```

## Contributing New Workflows

To add a new workflow, follow these steps:

1. Create a new directory under `src/comfy_models/workflows/` with your workflow name
2. Required files:

```
workflows/
└── your_workflow_name/
    ├── config.py          # Workflow configuration
    ├── runner.py          # Modal app definition
    ├── workflow.json      # ComfyUI workflow definition
    └── workflow_api.json  # API interface definition
```

### Configuration (config.py)

```python
from comfy_models.base.comfy_utils import Config

config = Config(
    name="your_workflow_name",
    models_to_cache=[
        "path/to/model1.safetensors",
        "path/to/model2.safetensors"
    ],
    warmup_workflow=True,  # Run workflow once on startup
    run_twice=False,       # Run workflow twice for testing
    nodes_to_preload=[]    # Nodes to initialize on startup
)
```

### Runner (runner.py)

```python
import uuid
from comfy_models.base.base_app import _ComfyDeployRunnerModelsDownloadOptimzedImports
import modal
from comfy_models.workflows.shared import get_configs
from comfy_models.workflows.your_workflow_name.config import config

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
    ComfyDeployRunner().run.remote({
        "prompt_id": str(uuid.uuid4()),
        "inputs": {
            "your_input": "your value"
        },
    })
```

### Workflow Files

- `workflow.json`: Export your workflow from ComfyUI
- `workflow_api.json`: Define the API interface for your workflow using ComfyDeploy nodes

## Example Workflows

Check out existing workflows in the `workflows` directory for reference:

- `flux_dev`: Flux model with development settings
- `flux_schnell`: Optimized Flux model
- `sd3_5_large`: Stable Diffusion 3.5 large model