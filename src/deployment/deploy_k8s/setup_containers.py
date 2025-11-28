import pulumi
import pulumi_kubernetes as k8s


def setup_containers(project, namespace, k8s_provider, ksa_name, app_name):
    # Get image references from deploy_images stack
    # For local backend, use: "organization/project/stack"
    # images_stack = pulumi.StackReference("organization/deploy-images/dev")

    # Get the image tags (these are arrays, so we take the first element)
    # NOTE: adjust these output names to match your deploy-images stack
    base_tag = "us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/"
    frontend_tag = base_tag + "frontend:latest"
    orchestrator_tag = base_tag + "orchestrator:latest"
    vlmodel_tag = base_tag + "bitefinder-vlmodel:latest"
    ragmodel_tag = base_tag + "ragmodel:latest"
    input_evaluation_tag = base_tag + "input-evaluation:latest"

    # General persistent storage for application data (used for VL model cache, 10Gi)
    vlmodel_cache_pvc = k8s.core.v1.PersistentVolumeClaim(
        "vlmodel-cache-pvc",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="vlmodel-cache-pvc",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
            access_modes=["ReadWriteOnce"],
            resources=k8s.core.v1.VolumeResourceRequirementsArgs(
                requests={"storage": "10Gi"},
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    # --- Frontend Deployment ---
    # Creates pods running the frontend container on port 3000
    frontend_deployment = k8s.apps.v1.Deployment(
        "frontend",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="frontend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "frontend"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "frontend"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="frontend",
                            image=frontend_tag,
                            image_pull_policy="IfNotPresent",
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=3000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                # Match docker-compose: NEXT_PUBLIC_BASE_API_URL=http://orchestrator:9000
                                k8s.core.v1.EnvVarArgs(
                                    name="NEXT_PUBLIC_BASE_API_URL",
                                    value="/orchestrator",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="PORT",
                                    value="3000",
                                ),
                                # CHOKIDAR_USEPOLLING is mainly for dev; keep for parity if desired
                                k8s.core.v1.EnvVarArgs(
                                    name="CHOKIDAR_USEPOLLING",
                                    value="true",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "2Gi"},
                                limits={"cpu": "500m", "memory": "3Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    frontend_service = k8s.core.v1.Service(
        "frontend-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="frontend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=3000,
                    target_port=3000,
                    protocol="TCP",
                )
            ],
            selector={"run": "frontend"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[frontend_deployment]),
    )

    # --- RAG Model Deployment (pinecone-based) ---
    ragmodel_deployment = k8s.apps.v1.Deployment(
        "ragmodel",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="ragmodel",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "ragmodel"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "ragmodel"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="ragmodel",
                            image=ragmodel_tag,
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=9000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GOOGLE_APPLICATION_CREDENTIALS",
                                    value="/secrets/bitefinder-service-account.json",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_PROJECT",
                                    value=project,
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="PINECONE_INDEX",
                                    value="bugbite-rag",  # from docker-compose
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "512Mi"},
                                limits={"cpu": "500m", "memory": "1Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    ragmodel_service = k8s.core.v1.Service(
        "ragmodel-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="ragmodel",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=9000,
                    target_port=9000,
                    protocol="TCP",
                )
            ],
            selector={"run": "ragmodel"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[ragmodel_deployment]),
    )

    # --- VL Model Deployment (bitefinder-vlmodel) ---
    vlmodel_deployment = k8s.apps.v1.Deployment(
        "vlmodel",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="vlmodel",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "vlmodel"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "vlmodel"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,
                    volumes=[
                        k8s.core.v1.VolumeArgs(
                            name="vlmodel-cache",
                            persistent_volume_claim=(
                                k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(
                                    claim_name=vlmodel_cache_pvc.metadata.name,
                                )
                            ),
                        )
                    ],
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="vlmodel",
                            image=vlmodel_tag,
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=9000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GOOGLE_APPLICATION_CREDENTIALS",
                                    value="/secrets/bitefinder-service-account.json",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_PROJECT",
                                    value=project,
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_BUCKET_NAME",
                                    value="bitefinder-data",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="WANDB_TEAM",
                                    value="bitefinder",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="WANDB_PROJECT",
                                    value="bitefinder-vl",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="MODEL_CACHE_DIR",
                                    value="/app/vlmodel_cache",
                                ),
                            ],
                            volume_mounts=[
                                k8s.core.v1.VolumeMountArgs(
                                    name="vlmodel-cache",
                                    mount_path="/app/vlmodel_cache",
                                )
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "500m", "memory": "4Gi"},
                                limits={"cpu": "1", "memory": "8Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace, vlmodel_cache_pvc]),
    )

    vlmodel_service = k8s.core.v1.Service(
        "vlmodel-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="vlmodel",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=9000,
                    target_port=9000,
                    protocol="TCP",
                )
            ],
            selector={"run": "vlmodel"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[vlmodel_deployment]),
    )

    # --- Input Evaluation Deployment ---
    input_eval_deployment = k8s.apps.v1.Deployment(
        "input-evaluation",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="input-evaluation",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "input-evaluation"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "input-evaluation"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="input-evaluation",
                            image=input_evaluation_tag,
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=9000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GOOGLE_APPLICATION_CREDENTIALS",
                                    value="/secrets/bitefinder-service-account.json",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "512Mi"},
                                limits={"cpu": "500m", "memory": "1Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    input_eval_service = k8s.core.v1.Service(
        "input-evaluation-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="input-evaluation",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=9000,
                    target_port=9000,
                    protocol="TCP",
                )
            ],
            selector={"run": "input-evaluation"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[input_eval_deployment]),
    )

    # --- Orchestrator Deployment (calls others) ---
    orchestrator_deployment = k8s.apps.v1.Deployment(
        "orchestrator",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="orchestrator",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "orchestrator"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "orchestrator"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="orchestrator",
                            image=orchestrator_tag,
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=9000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="RAG_MODEL_URL",
                                    value="http://ragmodel:9000",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="VL_MODEL_URL",
                                    value="http://vlmodel:9000",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="INPUT_EVAL_URL",
                                    value="http://input-evaluation:9000",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="ALLOW_ORIGINS",
                                    value="http://localhost:3001",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "1Gi"},
                                limits={"cpu": "500m", "memory": "2Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider,
            depends_on=[
                ragmodel_service,
                vlmodel_service,
                input_eval_service,
            ],
        ),
    )

    orchestrator_service = k8s.core.v1.Service(
        "orchestrator-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="orchestrator",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=9000,
                    target_port=9000,
                    protocol="TCP",
                )
            ],
            selector={"run": "orchestrator"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[orchestrator_deployment]),
    )

    # Keep the same pattern of returning frontend + main API-like service
    return frontend_service, orchestrator_service
