#!/bin/bash

echo "TODO: fill in the docker run command"

#!/bin/bash
docker run -d -p 8000:8000 -e COMET_API_KEY=${COMET_API_KEY} app:latest
printf "\n\n\n Done running serving \n\n\n"
docker run -d -p 8001:8001 -e COMET_API_KEY=${COMET_API_KEY} app-streamlit:latest