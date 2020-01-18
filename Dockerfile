FROM pytorch/pytorch
# copy files and install dependencies
COPY . .
RUN pip install -r requirements_docker.txt

EXPOSE 8000
ENTRYPOINT python3 server.py