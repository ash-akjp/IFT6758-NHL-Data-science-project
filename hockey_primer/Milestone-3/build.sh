#!/bin/bash

echo "TODO: fill in the docker build command"

#!/bin/bash
docker build --no-cache -t app:latest -f Dockerfile.serving . 
docker build --no-cache -t app-streamlit:latest -f Dockerfile.streamlit . 