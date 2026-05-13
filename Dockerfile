# syntax=docker/dockerfile:1.4

# Definition of Submission container
ARG DOCKER_REGISTRY=docker.io
ARG ARCH=amd64
ARG DISTRO=ente
ARG BASE_TAG=${DISTRO}-${ARCH}

FROM ${DOCKER_REGISTRY}/duckietown/challenge-aido_lf-template-pytorch:${BASE_TAG}

ARG PIP_INDEX_URL="https://pypi.org/simple/"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

COPY ./dependencies.txt ./dependencies.txt
RUN python3 -m pip install -r ./dependencies.txt

RUN rm -rf /workspace && mkdir /workspace
WORKDIR /workspace

COPY ./solution /workspace/solution
COPY ./training /workspace/training
COPY ./models /workspace/models

CMD ["python3", "-m", "solution.main"]
