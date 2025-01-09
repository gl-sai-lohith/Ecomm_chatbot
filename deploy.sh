#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Update System and Install Required Packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io git

# Step 2: Define Variables
GIT_REPO_URL="https://github.com/your-username/your-repo.git" # Replace with your Git repo URL
DOCKER_IMAGE_NAME="my-docker-image"  # Replace with your desired Docker image name
DOCKER_CONTAINER_NAME="my-container" # Replace with your desired container name
HOST_PORT=80                         # Replace with the port you want to expose
CONTAINER_PORT=80                    # Replace with the port exposed by the container

# Step 3: Clone the Git Repository
echo "Cloning repository: $GIT_REPO_URL..."
if [ -d "repo" ]; then
    echo "Repo folder already exists. Deleting it first."
    rm -rf repo
fi
git clone $GIT_REPO_URL repo
cd repo

# Step 4: Build the Docker Image
echo "Building Docker image: $DOCKER_IMAGE_NAME..."
sudo docker build -t $DOCKER_IMAGE_NAME .

# Step 5: Stop and Remove Existing Container (if exists)
echo "Stopping and removing any existing containers with name: $DOCKER_CONTAINER_NAME..."
if [ "$(sudo docker ps -aq -f name=$DOCKER_CONTAINER_NAME)" ]; then
    sudo docker stop $DOCKER_CONTAINER_NAME
    sudo docker rm $DOCKER_CONTAINER_NAME
fi

# Step 6: Run the Docker Container
echo "Deploying container: $DOCKER_CONTAINER_NAME..."
sudo docker run -d --name $DOCKER_CONTAINER_NAME -p $HOST_PORT:$CONTAINER_PORT $DOCKER_IMAGE_NAME

# Step 7: Confirm Deployment
echo "Deployment completed successfully!"
echo "Access your application at: http://$(curl -s ifconfig.me):$HOST_PORT"
