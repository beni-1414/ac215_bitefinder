import datetime
import os
from pathlib import Path

import pulumi
import pulumi_docker_build as docker_build
from pulumi import CustomTimeouts
from pulumi_gcp import artifactregistry  # noqa: F401  # kept for side effects/consistency

project = pulumi.Config("gcp").require("project")
location = os.environ["GCP_REGION"]
timestamp_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
repository_name = "bitefinder-images"
registry_url = f"us-east1-docker.pkg.dev/{project}/{repository_name}"


def _find_repo_root() -> Path:
    """
    Walk upward from this file to find the directory that contains our services
    (orchestrator/new-frontend/etc.), so paths work both locally and in CI images.
    """
    start = Path(__file__).resolve().parent
    for candidate in [start] + list(start.parents):
        if (candidate / "orchestrator").exists() and (candidate / "new-frontend").exists():
            return candidate
    return start


BASE_DIR = _find_repo_root()


def build_image(image_name: str, context_rel: str, dockerfile_name: str = "Dockerfile"):
    context_path = BASE_DIR / context_rel
    dockerfile_path = context_path / dockerfile_name

    image = docker_build.Image(
        f"build-{image_name}",
        tags=[pulumi.Output.concat(registry_url, "/", image_name, ":", timestamp_tag)],
        context=docker_build.BuildContextArgs(location=str(context_path)),
        dockerfile={"location": str(dockerfile_path)},
        platforms=[docker_build.Platform.LINUX_AMD64],
        push=True,
        opts=pulumi.ResourceOptions(
            custom_timeouts=CustomTimeouts(create="30m"),
            retain_on_delete=True,
        ),
    )
    pulumi.export(f"{image_name}-ref", image.ref)
    pulumi.export(f"{image_name}-tags", image.tags)
    return image


orchestrator_image = build_image("orchestrator", "orchestrator")
frontend_image = build_image("frontend", "new-frontend")
input_evaluation_image = build_image("input-evaluation", "input_evaluation")
ragmodel_image = build_image("ragmodel", "ragmodel")
# vlmodel_image = build_image("vlmodel", "vlmodel") # Too big, pushed once and refer to prebuilt image
