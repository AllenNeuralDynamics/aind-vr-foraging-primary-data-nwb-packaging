# hash:sha256:f879b62675d4f3e653c6803af1ab08495c5cc25a7a235ba6a9c0dd35893ff38f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.11.1 \
    pynwb==3.0.0 \
    -e git+https://github.com/AllenNeuralDynamics/Aind.Behavior.CoreAnalysis.git@688c4a0e660c97a892c5a85a9ff2d4dea09aeb67#egg=aind-behavior-core-analysis \
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
