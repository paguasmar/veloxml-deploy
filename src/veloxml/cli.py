import os
import sys
from pathlib import Path
import typer

from veloxml import __version__
from veloxml.config import VeloxConfig
from veloxml.orchestrator.skypilot import SkyPilotOrchestrator
from veloxml.packager.truss_driver import Packager
from veloxml.ui.console import (
    console,
    print_banner,
    print_step,
    print_success,
    print_error_box,
    print_warning,
    print_curl_box,
    print_services_table,
)

app = typer.Typer(
    name="veloxml",
    help="VeloxML ⚡ - Push to API in one command. Serverless ML deployment in your AWS/GCP account.",
    no_args_is_help=True,
    add_completion=False,
)

@app.command()
def init(
    name: str = typer.Argument("my-model-service", help="Name of your model service"),
    gpu: str = typer.Option(None, "--gpu", "-g", help="GPU accelerator type (e.g. T4:1, A10G:1, L4:1)"),
):
    """Initialize a lightweight, production-ready VeloxML model project."""
    print_banner()
    workdir = Path(".")
    
    print_step(1, 2, "Scaffolding model service", name)
    Packager.scaffold_starter_project(workdir, name)
    print_success("Created app.py (FastAPI inference template) & requirements.txt")
    
    print_step(2, 2, "Writing cloud configuration", "veloxml.yaml")
    config = VeloxConfig(name=name)
    if gpu:
        config.compute.accelerator = gpu
    config.save(workdir / "veloxml.yaml")
    print_success("Configuration saved to veloxml.yaml")
    
    console.print("\n[bold green]🚀 Ready to build and deploy:[/bold green]")
    console.print("  • Test locally:  [cyan]uvicorn app:app --reload --port 8000[/cyan]")
    console.print("  • Deploy to AWS: [bold cyan]veloxml deploy[/bold cyan]\n")

@app.command()
def deploy(
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and generate cloud specs without launching EC2 instances"),
    cloud: str = typer.Option("aws", "--cloud", help="Target cloud provider (aws/gcp)"),
    region: str = typer.Option("us-east-1", "--region", help="Cloud region"),
    gpu: str = typer.Option(None, "--gpu", "-g", help="Override GPU accelerator (e.g. T4:1, A10G:1)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed verbose logs"),
):
    """Deploy your model service to AWS/GCP and get an instant, production-ready cURL endpoint."""
    print_banner()
    workdir = Path(".")
    
    config_file = workdir / "veloxml.yaml"
    if not config_file.exists() and not (workdir / "app.py").exists():
        print_error_box(
            "Project Not Found",
            "No 'veloxml.yaml' or 'app.py' found in the current directory.",
            "Run `veloxml init` to create a starter project or add `app.py`."
        )
        raise typer.Exit(code=1)
        
    config = VeloxConfig.load_or_default(config_file)
    
    if cloud:
        config.compute.cloud = cloud
    if region:
        config.compute.region = region
    if gpu:
        config.compute.accelerator = gpu
        
    orchestrator = SkyPilotOrchestrator(config, workdir)
    
    try:
        endpoint = orchestrator.deploy(dry_run=dry_run, verbose=verbose)
        if endpoint:
            print_curl_box(
                service_name=config.name,
                endpoint_url=endpoint,
                sample_payload='{"prompt": "Deploying LLMs without DevOps friction!"}'
            )
    except Exception as e:
        if verbose:
            raise e
        print_error_box("Unexpected Error", str(e), "Use `veloxml deploy --verbose` to view stack trace.")
        raise typer.Exit(code=1)

@app.command()
def status():
    """Check the status of all active VeloxML services in your cloud account."""
    print_banner()
    services = SkyPilotOrchestrator.list_services()
    print_services_table(services)

@app.command()
def logs(
    name: str = typer.Argument(None, help="Service name to view logs for"),
    tail: int = typer.Option(100, "--tail", "-t", help="Number of trailing log lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow live log output stream"),
):
    """View application logs for a running VeloxML service."""
    print_banner()
    workdir = Path(".")
    config = VeloxConfig.load_or_default(workdir / "veloxml.yaml")
    service_name = name or config.name
    
    console.print(f"[bold cyan]▶ Fetching logs for service '[white]{service_name}[/white]'...[/bold cyan]\n")
    SkyPilotOrchestrator.get_service_logs(service_name, tail=tail, follow=follow)

@app.command()
def down(
    name: str = typer.Option(None, "--name", "-n", help="Service name to teardown"),
    purge_all: bool = typer.Option(False, "--all", "-a", help="Purge all controller nodes and lingering clusters to ensure $0 cost"),
):
    """Tear down cloud resources cleanly to stop incurring costs."""
    print_banner()
    workdir = Path(".")
    config = VeloxConfig.load_or_default(workdir / "veloxml.yaml")
    if name:
        config.name = name
        
    orchestrator = SkyPilotOrchestrator(config, workdir)
    orchestrator.teardown(purge_all=purge_all)

@app.command()
def version():
    """Display the VeloxML CLI version."""
    console.print(f"[bold cyan]VeloxML CLI[/bold cyan] version [bold green]{__version__}[/bold green]")

if __name__ == "__main__":
    app()
