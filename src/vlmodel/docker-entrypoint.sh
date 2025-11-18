#!/bin/bash
set -e

echo "Container is starting..."
echo "Architecture: $(uname -m)"

echo "Environment ready."
echo "Python version: $(python --version 2>/dev/null || echo 'python not found')"
echo "UV version: $(uv --version 2>/dev/null || echo 'uv not found')"

# If arguments are passed, execute them instead of starting the server
if [ $# -gt 0 ]; then
  echo "Executing command: $@"
  exec "$@"
fi

# Run the api.main:app FastAPI application (project package is 'app')
uvicorn_server() {
    uvicorn api.main:api --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir api/
}

uvicorn_server_production() {
    uvicorn api.main:api --host 0.0.0.0 --port 9000 --lifespan on
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