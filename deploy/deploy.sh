#!/bin/sh

set -e

REGISTRY_NAME="tlob1acr"
REGISTRY_URL="${REGISTRY_NAME}.azurecr.io"
IMAGE_NAME="tlo"
IMAGE_TAG="1.1"
IMAGE_FULL_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Documentation at
# https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli.

# Login to Azure Container Registry
echo -n "Logging into ${REGISTRY_NAME}..."
az acr login --name "${REGISTRY_NAME}"
echo "done"
# Build the image
echo "Building docker image ${IMAGE_FULL_NAME}..."
docker build --tag "${IMAGE_FULL_NAME}" .
# Tag the image
echo -n "Tagging ${REGISTRY_URL}/${IMAGE_FULL_NAME}..."
docker tag "${IMAGE_FULL_NAME}" "${REGISTRY_URL}/${IMAGE_FULL_NAME}"
echo "done"
# Push the image
echo "Pushing ${REGISTRY_URL}/${IMAGE_FULL_NAME}..."
docker push "${REGISTRY_URL}/${IMAGE_FULL_NAME}"
