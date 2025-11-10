#!/bin/bash
set -e

echo "Container is starting..."
echo "Architecture: $(uname -m)"

# Activate virtual environment (guarded)
VENV_PATH=/home/app/.venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment at $VENV_PATH"
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: virtualenv not found at $VENV_PATH. Continuing without activating venv." >&2
fi

echo "Environment ready." 
echo "Python version: $(python --version 2>/dev/null || echo 'python not found')"
echo "UV version: $(uv --version 2>/dev/null || echo 'uv not found')"

# Run the app.main:app FastAPI application (project package is 'app')
uvicorn_server() {
    uvicorn app.main:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir app/ "$@"
}

uvicorn_server_production() {
    uv run uvicorn app.main:app --host 0.0.0.0 --port 9000 --lifespan on
}

export -f uvicorn_server
export -f uvicorn_server_production

echo -en "\033[92m
The following commands are available:
    uvicorn_server
        Run the Uvicorn Server
\033[0m
"

if [ "${DEV}" = 1 ]; then
    # Development mode: Keep shell open so developer can run uvicorn manually
    exec /bin/bash
else
    # Production mode: Run server in the foreground
    uvicorn_server_production
fi