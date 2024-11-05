import asyncio
from collections import deque
import json
from datetime import datetime
import traceback
from typing import List, Optional, Union, cast

import aiohttp
from .docker import (
    extract_hash,
    extract_url,
    CustomNode,
    DepsBody,
    DockerStep,
    DockerSteps,
    generate_all_docker_commands,
)
import modal
from pydantic import BaseModel, Field, validator

COMFY_HOST = "127.0.0.1:8188"


class Input(BaseModel):
    prompt_id: str
    workflow_api: Optional[Union[dict, str]] = None
    auth_token: Optional[str] = None
    inputs: Optional[dict]
    workflow_api_raw: Optional[Union[dict, str]] = None
    status_endpoint: Optional[str] = None
    file_upload_endpoint: Optional[str] = None
    gpu_event_id: Optional[str] = None

    @validator("workflow_api_raw", pre=True)
    def parse_workflow_api_raw(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for workflow_api_raw")
        return v


async def preload_node(class_type: str, workflow_api_raw: str, prompt_executor):
    workflow_api = json.loads(workflow_api_raw)

    # Find all nodes with matching class_type
    matching_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow_api.items()
        if node_data.get("class_type") == class_type
    ]

    # Process each matching node
    for node_id, node_data in matching_nodes:
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs", {})

        # Dynamically import the nodes module and get the class
        import importlib
        import nodes

        # nodes_module = importlib.import_module("nodes")
        node_class = getattr(nodes, class_type)

        # Create instance of the dynamically imported class
        node_instance = node_class()

        # Get the input types from the class
        input_types = node_class.INPUT_TYPES()
        # Get the function name from the class
        function_name = node_class.FUNCTION

        # Get the actual function
        node_function = getattr(node_instance, function_name)

        # Process inputs according to their defined types
        processed_inputs = {}
        required_inputs = input_types.get("required", {})
        optional_inputs = input_types.get("optional", {})

        # Handle both required and optional inputs
        for input_name, input_value in inputs.items():
            input_type = required_inputs.get(input_name) or optional_inputs.get(
                input_name
            )
            if input_type:
                # Convert input value based on type if needed
                processed_inputs[input_name] = input_value

        # Call the function with processed inputs
        print(f"Loading {class_type} with inputs: {processed_inputs}")
        result = node_function(**processed_inputs)

        print(f"Loaded {class_type} with result: {result}")

        # import torch

        # if node_class == "UNETLoader":
        #     m = result[0].clone()
        #     m.add_object_patch("diffusion_model", torch.compile(model=m.get_model_object("diffusion_model"), backend="inductor"))
        #     result[0] = m

        prompt_executor.caches.outputs.set(node_id, [result])
        prompt_executor.caches.objects.set(node_id, node_instance)


async def start_comfy(logs=None):
    # cmd = "comfy launch --background"
    cmd = "python /comfyui/main.py --dont-print-server --enable-cors-header --listen --port 8188"
    print("Starting ComfyUI")
    # await asyncio.subprocess.create_subprocess_shell(cmd, shell=True)
    server_process = await asyncio.subprocess.create_subprocess_shell(
        cmd,
        shell=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_task = asyncio.create_task(
        read_stream(stream=server_process.stdout, logs=logs)
    )
    stderr_task = asyncio.create_task(
        read_stream(stream=server_process.stderr, isStderr=True, logs=logs)
    )

    async def cleanup():
        stdout_task.cancel()
        stderr_task.cancel()
        # server_process.terminate()
        # await server_process.wait()

    return cleanup


async def wait_for_server(
    url=f"http://{COMFY_HOST}", delay=50, logs=[], last_sent_log_index=-1
):
    """
    Checks if the API is reachable
    """
    import aiohttp

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    # If the response status code is 200, the server is up and running
                    if response.status == 200:
                        print("API is reachable")
                        # yield f"event: event_update\ndata: {json.dumps({'event': 'comfyui_api_ready'})}\n\n"
                        return

        except Exception as e:
            # If an exception occurs, the server may not be ready
            pass

        if logs and last_sent_log_index != -1:
            while last_sent_log_index < len(logs):
                log = logs[last_sent_log_index]
                if isinstance(log["timestamp"], float):
                    log["timestamp"] = (
                        datetime.utcfromtimestamp(log["timestamp"]).isoformat() + "Z"
                    )
                print(log)
                yield log
                # yield f"event: log_update\ndata: {json.dumps(log)}\n\n"
                last_sent_log_index += 1

        # Wait for the specified delay before retrying
        await asyncio.sleep(delay / 1000)


async def read_stream(
    stream, log_queues=None, cold_start_queue=None, logs=None, isStderr=False
):
    import time

    if stream is None:
        print(f"{'stderr' if isStderr else 'stdout'} stream is None")
        return

    while True:
        try:
            line = await stream.readline()
            if line:
                l = line.decode("utf-8").strip()

                if l == "":
                    continue

                # if log_queues is not None and len(log_queues) > 0:
                #     target_log = log_queues[0]["logs"]
                # else:
                #     target_log = cold_start_queue

                # target_log = cast(deque, target_log)
                # target_log.append({"logs": l, "timestamp": time.time()})
                # print("appending to log queue", len(target_log), target_log)

                if not isStderr:
                    print(l, flush=True)
                    logs.append({"logs": l, "timestamp": time.time()})
                else:
                    print(l, flush=True)
                    logs.append({"logs": l, "timestamp": time.time()})
            else:
                break
        except asyncio.CancelledError:
            # Handle the cancellation here if needed
            break  # Break out of the loop on cancellation


async def queue_workflow(data: Input):
    data_str = data.json()
    data_bytes = data_str.encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {data.auth_token}",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{COMFY_HOST}/comfyui-deploy/run", data=data_bytes, headers=headers
        ) as response:
            return await response.json()


async def check_status(prompt_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://{COMFY_HOST}/comfyui-deploy/check-status?prompt_id={prompt_id}"
        ) as response:
            return await response.json()


async def wait_for_completion(prompt_id: str):
    retries = 0
    status = ""
    try:
        while True:
            status_result = await check_status(prompt_id=prompt_id)
            if "status" in status_result and (
                status_result["status"] == "success"
                or status_result["status"] == "failed"
            ):
                status = status_result["status"]
                print(status)
                break
            else:
                # Wait before trying again
                await asyncio.sleep(250 / 1000)
                retries += 1
    except Exception as e:
        print(traceback.format_exc())
        raise e
        # return {"error": f"Error waiting for image generation: {str(e)}"}


def generate_modal_image(
    dependencies: Optional[Union[List[str], DepsBody]] = Field(
        None,
        description="The dependencies to use, either as a DepsBody or a list of shorthand strings",
        examples=[
            [
                "BennyKok/ComfyUI@d7c030b",
                "Stability-AI/ComfyUI-SAI_API@1793086",
                "cubiq/ComfyUI_IPAdapter_plus@b188a6c",
            ]
        ],
    ),
    link_models: bool = True,
    # comfyui_override: Optional[str] = Field(
    #     None,
    #     examples=[
    #         [
    #             "BennyKok/ComfyUI@d7c030b",
    #         ]
    #     ],
    # ),
):
    dockerfile_image = modal.Image.debian_slim(python_version="3.11")

    if dependencies:
        if isinstance(dependencies, list):
            # Handle shorthand dependencies
            steps = DockerSteps(
                steps=[
                    DockerStep(
                        type="custom-node",
                        data=CustomNode(
                            install_type="git-clone",
                            url=extract_url(dep),
                            hash=extract_hash(dep),
                            name=dep.split("/")[-1].split("@")[0],
                        ),
                    )
                    for dep in dependencies
                ]
            )
            # if comfyui_override is not None:
            #     steps.steps.append(
            #         CustomNode(
            #             install_type="git-clone",
            #             url=extract_url(comfyui_override),
            #             hash=extract_hash(comfyui_override),
            #             name="comfyui",
            #         )
            #     )
            deps_body = DepsBody(docker_command_steps=steps)
            converted = generate_all_docker_commands(deps_body)
        else:
            converted = generate_all_docker_commands(dependencies)

        docker_commands = converted.docker_commands
        if docker_commands is not None:
            for commands in docker_commands:
                dockerfile_image = dockerfile_image.dockerfile_commands(
                    commands,
                )

    if link_models:
        dockerfile_image = dockerfile_image.run_commands(
            "rm -rf /comfyui/models && ln -s  /private_models /comfyui/models",
        )

    dockerfile_image = dockerfile_image.env({"XFORMERS_ENABLE_TRITON": "1"})

    return dockerfile_image


def optimize_image(image: modal.Image):
    with image.imports():
        import torch
        import safetensors
        import xformers

        # import sys
        # from pathlib import Path

        # Add the ComfyUI directory to the Python path
        # comfy_path = Path("/comfyui")
        # sys.path.append(str(comfy_path))

        # # Add the ComfyUI/utils directory to the Python path
        # utils_path = comfy_path / "utils"
        # sys.path.append(str(utils_path))

        # import nodes
        # import server
        # import execution
        # import comfy

    return image


class Config(BaseModel):
    id: str
    name: str
    warmup_workflow: bool = False
    run_twice: bool = False
    preview_image: Optional[str] = None
    models_to_cache: List[str] = []
    nodes_to_preload: List[str] = []
    gpu: Optional[Union[str, List[str]]] = "a100"
