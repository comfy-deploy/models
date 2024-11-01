import uuid
import click
import modal

from comfy_models.workflows import get_all_workflow_configs
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
import subprocess
import os


def print_loading_times(title, loading_times):
    if not loading_times:  # Skip if no loading times are available
        return
    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Step':<35} | {'Time (seconds)':<15}")
    print("-" * 50)
    for step, t in loading_times.items():
        print(f"{step:<35} | {t:<15.2f}")
    print("-" * 50)
    print()


try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    pass

console = Console()
workflow_configs = get_all_workflow_configs()

# Get the base directory for workflows
# Assuming your workflows are in a 'workflows' directory at the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(BASE_DIR, "workflows")


def format_value_as_tree(value, tree):
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, dict):
                subtree = tree.add(f"[cyan]Item {i}")
                for k, v in item.items():
                    format_value_as_tree(v, subtree.add(f"[yellow]{k}"))
            else:
                tree.add(str(item))
    elif isinstance(value, dict):
        for k, v in value.items():
            format_value_as_tree(v, tree.add(f"[yellow]{k}"))
    else:
        tree.label = f"{tree.label}: [green]{str(value)}"


@click.group()
def cli():
    """ComfyUI Workflow Management CLI"""
    pass


@click.command()
def ls():
    """List all available workflows with summary"""
    table = Table(title="Available Workflows")
    table.add_column("Workflow", style="cyan")
    table.add_column("Inputs", style="green")
    table.add_column("Outputs", style="yellow")

    for name, config in workflow_configs.items():
        config_dict = config.dict()
        inputs = [
            f"{inp['input_id']} ({inp['class_type']})"
            for inp in config_dict.get("inputs", [])
        ]
        outputs = [
            f"{out['output_id']} ({out['class_type']})"
            for out in config_dict.get("outputs", [])
        ]
        table.add_row(name, "\n".join(inputs) or "None", "\n".join(outputs) or "None")

    console.print(table)


@click.command()
@click.argument("workflow_name")
def cat(workflow_name):
    """Show detailed tree view of a specific workflow"""
    if workflow_name.lower() == "all":
        for name, config in workflow_configs.items():
            _print_workflow_tree(name, config)
    elif workflow_name in workflow_configs:
        _print_workflow_tree(workflow_name, workflow_configs[workflow_name])
    else:
        console.print(f"[red]Workflow '{workflow_name}' not found[/red]")


def _print_workflow_tree(name, config):
    config_dict = config.dict()
    tree = Tree(f"[bold]{name}")
    for key, value in config_dict.items():
        subtree = tree.add(f"[yellow]{key}")
        format_value_as_tree(value, subtree)
    console.print(tree)
    console.print("\n")


@click.command()
@click.argument("workflow_name")
def deploy(workflow_name):
    """Deploy a specific workflow using modal"""
    if workflow_name.lower() == "all":
        for name in workflow_configs.keys():
            console.print(f"\n[bold]Deploying {name}...[/bold]")
            _deploy_workflow(name)
    elif workflow_name in workflow_configs:
        _deploy_workflow(workflow_name)
    else:
        console.print(f"[red]Workflow '{workflow_name}' not found[/red]")


def _deploy_workflow(workflow_name):
    """Helper function to deploy a single workflow"""
    workflow_dir = os.path.join(WORKFLOWS_DIR, workflow_name)
    try:
        subprocess.run(["modal", "deploy", "runner.py"], cwd=workflow_dir, check=True)
        console.print(f"[green]Successfully deployed {workflow_name}[/green]")
    except subprocess.CalledProcessError:
        console.print(f"[red]Error deploying {workflow_name}[/red]")
    except FileNotFoundError:
        console.print(f"[red]Workflow directory not found: {workflow_dir}[/red]")


@click.command()
@click.argument("workflow_name")
def run(workflow_name):
    """Run a specific workflow using modal"""
    if workflow_name in workflow_configs:
        workflow_dir = os.path.join(WORKFLOWS_DIR, workflow_name)
        try:
            subprocess.run(["modal", "run", "runner.py"], cwd=workflow_dir, check=True)
            console.print(f"[green]Successfully ran {workflow_name}[/green]")
        except subprocess.CalledProcessError:
            console.print(f"[red]Error running {workflow_name}[/red]")
        except FileNotFoundError:
            console.print(f"[red]Workflow directory not found: {workflow_dir}[/red]")
    else:
        console.print(f"[red]Workflow '{workflow_name}' not found[/red]")


@click.command()
@click.argument("workflow_name")
def test(workflow_name):
    """Run a specific test case"""
    if workflow_name in workflow_configs:
        config = workflow_configs[workflow_name]

        inputs = {}
        for input_config in config.inputs:
            if input_config.default_value is not None:
                inputs[input_config.input_id] = input_config.default_value

        # Create and display inputs table
        input_table = Table(title=f"Running workflow: {workflow_name}")
        input_table.add_column("Input ID", style="cyan")
        input_table.add_column("Value", style="green")

        for input_id, value in inputs.items():
            input_table.add_row(input_id, str(value))

        console.print(input_table)

        runner = modal.Cls.lookup(workflow_name, "ComfyDeployRunner")
        with modal.enable_output():
            result = runner.run.remote(
                {"prompt_id": str(uuid.uuid4()), "inputs": inputs}
            )
        print_loading_times(
            f"{workflow_name}",
            result.get("loading_time", {}),
        )
    else:
        console.print(f"[red]Workflow '{workflow_name}' not found[/red]")


# Register commands
cli.add_command(ls)
cli.add_command(cat)
cli.add_command(deploy)
cli.add_command(run)
cli.add_command(test)

if __name__ == "__main__":
    cli()
