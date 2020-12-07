FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget &&\
    rm -rf /var/lib/apt/lists/*


# Set up UID not to use root in Container
ARG UID
ARG GID
ARG UNAME

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}

RUN groupadd -g ${GID} ${UNAME}
RUN useradd -u ${UID} -g ${UNAME} -m ${UNAME}

ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:${PATH}
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh
RUN conda install -y conda

RUN conda install pytorch==1.7.0 cudatoolkit=11.0 torchvision==0.8.1 -c pytorch && \
    conda install -c anaconda scikit-learn==0.23.2 && \
    conda install -c conda-forge matplotlib==3.3.2 mlflow==1.11.0 tensorboard==1.15.0 && \
    conda clean -i -t -y

RUN pip install -U pip && \
    pip install --no-cache-dir hydra-core==0.11.3 torchsummary

WORKDIR /res
