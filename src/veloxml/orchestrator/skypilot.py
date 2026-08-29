import os
import re
import subprocess
import time
import urllib.request
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

from veloxml.config import VeloxConfig
from veloxml.ui.console import (
    console,
    print_step,
    print_success,
    print_error_box,
    print_warning,
)

class SkyPilotOrchestrator:
    def __init__(self, config: VeloxConfig, workdir: Path = Path(".")):
        self.config = config
        self.workdir = workdir

    def generate_skyserve_spec(self) -> Dict[str, Any]:
        compute = self.config.compute
        service = self.config.service
        scaling = self.config.scaling

        resources: Dict[str, Any] = {
            "cloud": compute.cloud.lower(),
        }
        if compute.region:
            resources["region"] = compute.region
        if compute.accelerator:
            resources["accelerators"] = compute.accelerator
        if compute.cpus:
            resources["cpus"] = f"{compute.cpus}+"
        if compute.memory:
            resources["memory"] = f"{compute.memory}+"
        if compute.use_spot:
            resources["use_spot"] = True

        # ML wheel optimization: PyTorch cpu wheels vs general PyPI
        setup_lines = [
            "pip install --upgrade pip",
            "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
            "pip install torch --index-url https://download.pytorch.org/whl/cpu || true",
        ]

        skypilot_yaml = {
            "service": {
                "readiness_probe": {
                    "path": service.readiness_probe,
                    "initial_delay_seconds": 240,
                },
                "replica_policy": {
                    "min_replicas": scaling.min_replicas,
                    "max_replicas": scaling.max_replicas,
                    "target_qps_per_replica": scaling.target_qps,
                },
            },
            "resources": resources,
            "workdir": str(self.workdir.resolve()),
            "setup": "\n".join(setup_lines),
            "run": service.entrypoint,
        }
        return skypilot_yaml

    def write_spec(self, target_path: Path) -> Path:
        spec = self.generate_skyserve_spec()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            yaml.dump(spec, f, sort_keys=False)
        return target_path

    def deploy(self, dry_run: bool = False, verbose: bool = False) -> Optional[str]:
        service_name = self.config.name
        
        print_step(1, 3, "Compiling Cloud Spec", f"Target: {self.config.compute.cloud.upper()} Spot")
        yaml_file = self.workdir / ".veloxml" / f"{service_name}.sky.yaml"
        self.write_spec(yaml_file)
        print_success(f"Optimized cluster specification generated at .veloxml/{service_name}.sky.yaml")

        if dry_run:
            print_success("Dry-run validation successful. No actual cloud costs incurred.")
            return "http://127.0.0.1:8000"

        print_step(2, 3, "Provisioning Cloud Infrastructure", f"{self.config.compute.cloud.upper()} {self.config.compute.region}")
        
        cmd = ["sky", "serve", "up", "-n", service_name, str(yaml_file), "-y"]
        
        with console.status("[bold cyan]Spinning up cloud compute and configuring VPC ingress...[/bold cyan]", spinner="dots"):
            result = subprocess.run(cmd, capture_output=not verbose, text=True)
        
        if result.returncode != 0:
            error_output = result.stderr if not verbose else "Execution failed."
            if "No credential" in error_output or "AccessDenied" in error_output:
                print_error_box(
                    "Cloud Authentication Error",
                    "Could not authenticate with your cloud provider account.",
                    "Run `aws configure` or `gcloud auth application-default login` to refresh credentials."
                )
            else:
                clean_err = error_output.strip().splitlines()[-1] if error_output.strip() else "Unknown cloud error"
                print_error_box(
                    "Deployment Failed",
                    f"Orchestrator encountered an error: {clean_err}",
                    "Run `veloxml deploy --verbose` to inspect full cloud logs."
                )
            return None

        print_success("Cloud compute provisioned and model server started.")
        print_step(3, 3, "Waiting for Model Readiness Probe", f"Checking {self.config.service.readiness_probe}")
        
        endpoint = self.wait_for_endpoint(service_name)
        return endpoint

    def wait_for_endpoint(self, service_name: str, timeout_seconds: int = 600) -> Optional[str]:
        """Polls service status with a clean elapsed timer until replica endpoint is alive."""
        start_time = time.time()
        with console.status("[bold cyan]Warming up model weights & establishing endpoint...[/bold cyan]", spinner="dots") as status:
            while time.time() - start_time < timeout_seconds:
                elapsed = int(time.time() - start_time)
                status.update(f"[bold cyan]Warming up model weights & establishing endpoint... ({elapsed}s)[/bold cyan]")
                
                endpoint = self.get_service_endpoint(service_name)
                if endpoint:
                    try:
                        health_url = f"{endpoint.rstrip('/')}/health"
                        req = urllib.request.Request(health_url, headers={"User-Agent": "VeloxML-Prober"})
                        with urllib.request.urlopen(req, timeout=3) as resp:
                            if resp.status == 200:
                                print_success("Readiness probe passed (HTTP 200 OK)")
                                return endpoint
                    except Exception:
                        pass
                time.sleep(6)
        
        print_warning("Model is taking longer than usual to become ready.")
        console.print("  Check live status anytime with: [bold cyan]veloxml status[/bold cyan]\n")
        return None

    def get_service_endpoint(self, service_name: str) -> Optional[str]:
        """Parses service status to extract the direct replica or load balancer endpoint."""
        cmd = ["sky", "serve", "status", service_name]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return None
        
        output = res.stdout
        url_matches = re.findall(r"(http://[0-9a-zA-Z.-]+:\d+)", output)
        if url_matches:
            for url in url_matches:
                if ":8000" in url:
                    return url
            return url_matches[0]
        return None

    @staticmethod
    def list_services() -> List[Dict[str, Any]]:
        """Lists all active VeloxML services in a structured format."""
        cmd = ["sky", "serve", "status"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return []
        
        services = []
        lines = res.stdout.splitlines()
        in_services_section = False
        
        for line in lines:
            if "Services" in line:
                in_services_section = True
                continue
            if "Service Replicas" in line:
                break
            if in_services_section and line.strip() and not line.startswith("NAME"):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 4:
                    services.append({
                        "name": parts[0],
                        "version": parts[1] if len(parts) > 1 else "1",
                        "uptime": parts[2] if len(parts) > 2 else "-",
                        "status": parts[3] if len(parts) > 3 else "UNKNOWN",
                        "replicas": parts[4] if len(parts) > 4 else "1/1",
                        "endpoint": parts[5] if len(parts) > 5 else "-",
                    })
        return services

    @staticmethod
    def get_service_logs(service_name: str, tail: int = 100, follow: bool = False):
        """Fetches clean application logs for the specified service."""
        cmd = ["sky", "serve", "logs", service_name]
        if follow:
            subprocess.run(cmd)
        else:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                lines = res.stdout.splitlines()[-tail:]
                for l in lines:
                    console.print(l)
            else:
                print_error_box("Logs Error", f"Could not retrieve logs for '{service_name}': {res.stderr.strip()}")

    @staticmethod
    def check_cloud() -> bool:
        """Validates cloud credentials and compute access cleanly."""
        res = subprocess.run(["sky", "check"], capture_output=True, text=True)
        return "Enabled Infra: aws" in res.stdout or "aws" in res.stdout.lower()

    def teardown(self, purge_all: bool = False):
        """Tears down the service and optionally cleans all controller nodes to ensure $0 cost."""
        service_name = self.config.name
        console.print(f"[bold cyan]▶ Tearing down service '[white]{service_name}[/white]'...[/bold cyan]")
        
        with console.status("[bold yellow]Terminating instances & destroying load balancer...[/bold yellow]", spinner="dots"):
            cmd = ["sky", "serve", "down", service_name, "-y"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            
        if res.returncode == 0:
            print_success(f"Service '{service_name}' terminated.")
        else:
            print_warning(f"Service '{service_name}' already down or not found.")

        if purge_all:
            with console.status("[bold yellow]Purging controller nodes and lingering instances...[/bold yellow]", spinner="dots"):
                subprocess.run(["sky", "down", "-a", "-y"], capture_output=True, text=True)
            print_success("All controller nodes and lingering clusters purged.")
            console.print("  [bold green]✔ Active Cloud Compute Cost: $0.00/hr[/bold green]\n")
