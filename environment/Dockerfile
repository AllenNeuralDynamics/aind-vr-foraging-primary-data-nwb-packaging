# hash:sha256:e441bb34a6c3edf552be4feeb26b4d49745286329e54974df15067ef5fc464f7
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
    -e git+https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.git@feat-add-data-contract#egg=aind-behavior-vr-foraging \
    -e git+https://github.com/rly/ndx-events.git@0d4e8efdcf1ab2eb9da9ca078b4522fcb82fa8a6#egg=ndx-events

ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server  /usr/bin/code-server

RUN mkdir -p /.vscode/extensions \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension REditorSupport.R \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.bioSyntax \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev
