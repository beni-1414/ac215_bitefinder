@echo off

SET IMAGE_NAME=synthetic-label-generation

REM Build Docker image
echo Building Docker image: %IMAGE_NAME%
docker build -t %IMAGE_NAME% -f Dockerfile .

REM Run Docker container with mounted volumes
echo Running Docker container: %IMAGE_NAME%
docker run --rm --name %IMAGE_NAME% -ti ^
    -v "%cd%:/app" ^
    --env-file ../../../.env ^
    %IMAGE_NAME%