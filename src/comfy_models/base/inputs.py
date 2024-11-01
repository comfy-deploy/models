import json
from typing import Any, Dict, List, Optional, Union

custom_input_nodes: Dict[str, Dict[str, str]] = {
    "ComfyUIDeployExternalText": {
        "type": "string",
        "description": "Multi-line text input",
    },
    "ComfyUIDeployExternalTextAny": {"type": "string", "description": "Any text input"},
    "ComfyUIDeployExternalTextSingleLine": {
        "type": "string",
        "description": "Single-line text input",
    },
    "ComfyUIDeployExternalImage": {"type": "string", "description": "Public image URL"},
    "ComfyUIDeployExternalImageAlpha": {
        "type": "string",
        "description": "Public image URL with alpha channel",
    },
    "ComfyUIDeployExternalNumber": {
        "type": "float",
        "description": "Floating-point number input",
    },
    "ComfyUIDeployExternalNumberInt": {
        "type": "integer",
        "description": "Integer number input",
    },
    "ComfyUIDeployExternalLora": {
        "type": "string",
        "description": "Public LoRA download URL",
    },
    "ComfyUIDeployExternalCheckpoint": {
        "type": "string",
        "description": "Public checkpoint download URL",
    },
    "ComfyDeployWebscoketImageInput": {
        "type": "binary",
        "description": "Websocket image input",
    },
    "ComfyUIDeployExternalImageBatch": {
        "type": "string",
        "description": "Array of image URLs",
    },
    "ComfyUIDeployExternalVideo": {"type": "string", "description": "Public video URL"},
    "ComfyUIDeployExternalBoolean": {"type": "boolean", "description": "Boolean input"},
    "ComfyUIDeployExternalNumberSlider": {
        "type": "float",
        "description": "Floating-point number slider",
    },
    "ComfyUIDeployExternalNumberSliderInt": {
        "type": "integer",
        "description": "Integer number slider",
    },
    "ComfyUIDeployExternalEnum": {
        "type": "string",
        "description": "Enumerated string options",
    },
}

# This is a type hint for the CustomInputNodesTypeMap
CustomInputNodesTypeMap = Dict[str, Union[str, int, float, bool, List[str], bytes]]

# Define InputsType as a string literal type (closest Python equivalent)
InputsType = str

# Create the list of input types
input_types_list: List[InputsType] = list(custom_input_nodes.keys())


def get_inputs_from_workflow_api(
    workflow_api: Optional[Union[str, Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if not workflow_api:
        return None
    
    if isinstance(workflow_api, str):
        workflow_api = json.loads(workflow_api)

    inputs = []
    for _, value in workflow_api.items():
        if not value.get("class_type"):
            continue

        node_type = custom_input_nodes.get(value["class_type"])

        if node_type:
            input_id = value["inputs"].get("input_id", "")
            default_value = value["inputs"].get("default_value")

            input_data = {
                **value["inputs"],
                "class_type": value["class_type"],
                "type": node_type.get("type"),
                "input_id": input_id,
                "default_value": default_value,
                "min_value": value["inputs"].get("min_value"),
                "max_value": value["inputs"].get("max_value"),
                "display_name": value["inputs"].get("display_name", ""),
                "description": value["inputs"].get("description", ""),
            }
            inputs.append(input_data)

    return inputs if inputs else []
