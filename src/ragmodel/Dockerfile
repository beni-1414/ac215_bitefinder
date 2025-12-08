# Base image
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential curl"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# uv settings
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app && \
    mkdir -p /.venv && \
    chown app:app /.venv

USER app
WORKDIR /app

# Copy dependency files
COPY --chown=app:app pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen

# Copy the source code
COPY --chown=app:app . .

# Add the Docker entrypoint
COPY --chown=app:app docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose the service port
EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
