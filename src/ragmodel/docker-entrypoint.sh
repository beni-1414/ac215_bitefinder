#!/bin/bash
set -e

echo "RAG Service container starting..."
echo "Python: $(python --version 2>/dev/null || echo 'python missing')"
echo "UV: $(uv --version 2>/dev/null || echo 'uv missing')"

# Activate uv-managed virtual environment
echo "Activating /.venv..."
source /.venv/bin/activate

# If user passed a command, run it instead
if [ $# -gt 0 ]; then
  echo "Executing custom command: $@"
  exec "$@"
fi

uvicorn_server_dev() {
    uvicorn api.main:api \
        --host 0.0.0.0 \
        --port 9000 \
        --log-level debug \
        --reload \
        --reload-dir api/
}

uvicorn_server_prod() {
    uvicorn api.main:api \
        --host 0.0.0.0 \
        --port 8080 \
        --lifespan on
}

export -f uvicorn_server_dev
export -f uvicorn_server_prod

if [ "${DEV}" = 1 ]; then
    echo "Launching in DEV mode..."
    exec /bin/bash
else
    echo "Launching in PRODUCTION mode..."
    uvicorn_server_prod
fi
