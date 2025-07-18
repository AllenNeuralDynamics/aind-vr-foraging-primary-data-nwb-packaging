# hash:sha256:893bdd29307880ab37a1d5088b02123ae2b5bd87b49661eb1a3f93f7f991f555
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
        libglib2.0-0=2.72.4-0ubuntu2.5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    contraqctor==0.4.1 \
    hdmf-zarr==0.11.1 \
    pydantic-settings==2.9.1 \
    pynwb==3.0.0 \
    -e git+https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.git#egg=aind-behavior-vr-foraging \
    -e git+https://github.com/rly/ndx-events.git@0d4e8efdcf1ab2eb9da9ca078b4522fcb82fa8a6#egg=ndx-events
