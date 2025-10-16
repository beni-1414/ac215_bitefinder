# use the official debian-hosted python image
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential curl"
# install small runtime libs that pillow/torchvision sometimes use
ARG RUNTIME_LIBS="libjpeg62-turbo libpng16-16 libgl1"

# prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive
# utf-8 locale
ENV LANG=C.UTF-8
# unbuffered logs
ENV PYTHONUNBUFFERED=1

# uv settings to keep a project venv at /.venv
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# ensure we pull cpu wheels for torch/torchvision by default
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# baseline, deps, uv, and an unprivileged user
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES $RUNTIME_LIBS && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app /.venv && \
    chown -R app:app /app /.venv

# switch to non-root user
USER app
WORKDIR /app

# copy source
COPY --chown=app:app . ./

# install python deps into /.venv from pyproject.toml / uv.lock
RUN uv sync

# entry point and default command: drop into an activated shell
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "source /.venv/bin/activate && exec bash"]
