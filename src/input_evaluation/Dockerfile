# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Tell uv to copy packages from the wheel into the site-packages
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv

# Ensure we have an up to date baseline, install dependencies and
# create a user so we don't run the app as root
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
    mkdir -p /home/app/.venv && \
    chown app:app /home/app/.venv

# Switch to the new user
USER app
WORKDIR /app

# Copy the source code
COPY --chown=app:app . ./

# Copy the repository entrypoint into a location that won't be
# overwritten by a host bind-mount (the build's /app is later
# replaced by the host's /app when you run with -v "$(pwd)":/app).
# Place it in /home/app, make it executable, and keep ownership.
RUN cp /app/docker-entrypoint.sh /home/app/docker-entrypoint.sh && \
    chmod +x /home/app/docker-entrypoint.sh && \
    chown app:app /home/app/docker-entrypoint.sh && \
    uv sync

# Use the stable entrypoint path that is not affected by mounting /app
ENTRYPOINT ["/home/app/docker-entrypoint.sh"]