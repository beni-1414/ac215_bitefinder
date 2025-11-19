# ----------------------
# Stage 1: Builder
# ----------------------
FROM python:3.12-slim-bookworm AS builder

ARG DEBIAN_PACKAGES="build-essential curl"

# Prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install build tools, pip, uv
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    mkdir -p /.venv && \
    chmod 777 /.venv

WORKDIR /app

# Copy dependency files and install packages in virtualenv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen

# Copy app code
COPY api/ ./api/
COPY tests/ ./tests/
COPY pytest.ini ./
COPY docker-entrypoint.sh ./

# ----------------------
# Stage 2: Runtime
# ----------------------
FROM python:3.12-slim-bookworm

# Minimal environment
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_LINK_MODE=copy
ENV PATH="/.venv/bin:${PATH}"

# Create app user
RUN useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app && \
    mkdir -p /.venv && \
    chown app:app /.venv

USER app
WORKDIR /app

# Copy virtualenv + app from builder
COPY --from=builder /.venv /.venv
COPY --from=builder /app/api ./api
COPY --from=builder /app/tests ./tests
COPY --from=builder /app/pytest.ini ./
COPY --from=builder /app/docker-entrypoint.sh ./

# Make entrypoint executable
USER root
RUN chmod +x docker-entrypoint.sh
USER app

# Expose service port
EXPOSE 9000

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
