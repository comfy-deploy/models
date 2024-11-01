from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

comfydeploy_hash = "7fdfba6b6e617a14d663ac78e6f27a517d2b3fe6"
comfyui_hash = "8ce2a1052ca03183768da0aaa483024e58b8008c"


def extract_hash(dependency_string):
    parts = dependency_string.split("@")
    if len(parts) > 1:
        return parts[-1]
    return ""


def extract_url(dependency_string):
    parts = dependency_string.split("@")
    if len(parts) > 1:
        return "https://github.com/" + parts[0]
    return ""


def comfyui_cmd(
    cpu: bool = False,
    extra_args: Optional[str] = None,
):
    cmd = (
        f"python main.py --dont-print-server --enable-cors-header --listen --port 8188"
    )
    if cpu:
        cmd += " --cpu"
    if extra_args is not None:
        cmd += f" {extra_args}"

    print("Actual file command: ", cmd)

    return cmd


class CustomNode(BaseModel):
    pip: Optional[List[str]] = None
    url: str
    hash: Optional[str] = None
    install_type: str
    files: Optional[List[str]] = None
    name: Optional[str] = None


class DependencyGraph(BaseModel):
    comfyui: str
    models: Dict[str, str]
    missing_nodes: List[str]
    custom_nodes: Dict[str, CustomNode]
    files: Dict[str, str]


class DockerStep(BaseModel):
    type: str
    data: Union[CustomNode, str]


class DockerSteps(BaseModel):
    steps: List[DockerStep]


def generate_docker_commands_for_custom_node(
    custom_node: CustomNode, custom_node_path: str = "/comfyui/custom_nodes"
) -> List[str]:
    commands = []
    commands.append(f"WORKDIR {custom_node_path}")

    if custom_node.pip:
        commands.append(f"RUN python -m pip install {' '.join(custom_node.pip)}")

    if custom_node.install_type == "git-clone":
        commands.append(f"RUN git clone {custom_node.url} --recursive")
        folder_name = custom_node.url.split("/")[-1].replace(".git", "")
        commands.append(f"WORKDIR {custom_node_path}/{folder_name}")
        if custom_node.hash:
            commands.append(f"RUN git reset --hard {custom_node.hash}")
        commands.append(
            "RUN if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi"
        )
        commands.append(
            "RUN if [ -f install.py ]; then python install.py || echo 'install script failed'; fi"
        )
    elif custom_node.install_type == "copy":
        if custom_node.files:
            for file_url in custom_node.files:
                if file_url.endswith("/"):
                    file_url = file_url[:-1]
                if file_url.endswith(".py"):
                    commands.append(f"RUN wget {file_url} -P .")

    return commands


def generate_docker_def_from_docker_steps(
    steps: DockerSteps, custom_node_path: str = "/comfyui/custom_nodes"
) -> List[List[str]]:
    def_list = []
    for step in steps.steps:
        if step.type == "custom-node":
            def_list.append(
                generate_docker_commands_for_custom_node(step.data, custom_node_path)
            )
        elif step.type == "commands":
            def_list.append(step.data.split("\n"))
    return def_list


def generate_docker_def_for_custom_nodes(
    custom_nodes_deps: Dict[str, CustomNode],
    custom_node_path: str = "/comfyui/custom_nodes",
) -> List[List[str]]:
    return [
        generate_docker_commands_for_custom_node(custom_node, custom_node_path)
        for custom_node in custom_nodes_deps.values()
    ]


def generate_docker_def_for_external_files(files: Dict[str, str]) -> List[List[str]]:
    # Implement this function if needed
    return []


def generate_docker_commands(deps: DependencyGraph) -> List[List[str]]:
    return generate_docker_def_for_custom_nodes(
        deps.custom_nodes
    ) + generate_docker_def_for_external_files(deps.files)


def generate_docker_commands_from_docker_steps(steps: DockerSteps) -> List[List[str]]:
    return generate_docker_def_from_docker_steps(steps)


class DepsBody(BaseModel):
    docker_command_steps: Any
    dependencies: Optional[DependencyGraph] = None
    snapshot: Optional[Dict[str, Any]] = None
    comfyui_version: str = comfyui_hash
    extra_docker_commands: Optional[List[Dict[str, str]]] = None


class DockerCommandResponse(BaseModel):
    docker_commands: List[List[str]]
    # deps: DependencyGraph


def generate_all_docker_commands(data: DepsBody) -> DockerCommandResponse:
    deps = data.dependencies
    docker_commands = None
    steps = data.docker_command_steps

    if steps:
        steps = DockerSteps(**steps) if isinstance(steps, dict) else steps
        comfy_ui_index = next(
            (
                i
                for i, step in enumerate(steps.steps)
                if step.type == "custom-node" and step.data.name.lower() == "comfyui"
            ),
            -1,
        )
        comfy_ui_override = None
        if comfy_ui_index != -1:
            comfy_ui_override = steps.steps.pop(comfy_ui_index)
        # print("steps", steps)
        # print("comfy_ui_override", comfy_ui_override)
        docker_commands = generate_docker_commands_from_docker_steps(steps)
    elif deps:
        docker_commands = generate_docker_commands(DependencyGraph(**deps))

    if not deps and data.snapshot:
        snapshot = data.snapshot
        deps = DependencyGraph(
            comfyui=snapshot["comfyui"],
            models={},
            missing_nodes=[],
            custom_nodes={
                key: CustomNode(
                    hash=value["hash"],
                    url=key,
                    name=key,
                    pip=value.get("pip"),
                    install_type="git-clone",
                )
                for key, value in snapshot["git_custom_nodes"].items()
            },
            files={},
        )
        docker_commands = generate_docker_commands(deps)

    if data.comfyui_version:
        if not deps:
            deps = DependencyGraph(
                comfyui=data.comfyui_version,
                models={},
                missing_nodes=[],
                custom_nodes={},
                files={},
            )
        else:
            deps.comfyui = data.comfyui_version

    if docker_commands and data.extra_docker_commands:
        for extra_command in data.extra_docker_commands:
            if extra_command["when"] == "before":
                docker_commands.insert(0, extra_command["commands"])
            else:
                docker_commands.append(extra_command["commands"])

    # if not docker_commands:
    #     raise ValueError("No docker commands")

    enable_uv = False
    docker_commands = [
        [
            y.replace("python -m pip install", "uv pip install") if enable_uv else y
            for y in x
        ]
        for x in docker_commands
    ]

    docker_commands = [
        [
            "RUN python --version",
            "RUN apt-get update && apt-get install -y git wget curl",
        ],
        ["RUN apt-get install -y libgl1-mesa-glx libglib2.0-0"],
        ["RUN python -m pip install aioboto3"],
        ["RUN pip freeze"],
        [
            f"RUN git clone {comfy_ui_override.data.url if comfy_ui_override else 'https://github.com/comfyanonymous/ComfyUI.git'} /comfyui",
            f"RUN cd /comfyui && git reset --hard {comfy_ui_override.data.hash if comfy_ui_override else deps.comfyui}",
        ],
        ["RUN pip freeze"],
        [
            "WORKDIR /comfyui",
            "RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121",
            "RUN python -m pip install xformers>=0.0.28",
            "RUN python -m pip install -r requirements.txt",
        ],
        ["RUN pip freeze"],
    ] + docker_commands

    if not any(
        step.type == "custom-node"
        and step.data.url.lower().startswith(
            "https://github.com/bennykok/comfyui-deploy"
        )
        for step in steps.steps
    ):
        docker_commands.append(
            [
                "WORKDIR /comfyui/custom_nodes",
                "RUN git clone https://github.com/bennykok/comfyui-deploy --recursive",
                "WORKDIR /comfyui/custom_nodes/comfyui-deploy",
                f"RUN git reset --hard {comfydeploy_hash}",
                "RUN if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi",
                "RUN if [ -f install.py ]; then python install.py || echo 'install script failed'; fi",
            ]
        )

    # return {
    #     "docker_commands": docker_commands,
    #     # "deps": deps
    # }

    return DockerCommandResponse(docker_commands=docker_commands)
