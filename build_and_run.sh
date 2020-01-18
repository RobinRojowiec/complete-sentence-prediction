#!/usr/bin/env bash

# build image
docker build . -t iscomp-api

# run image detached
docker run -d -p 8000:8000 iscomp-api

