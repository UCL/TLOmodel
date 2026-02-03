FROM python:3.11-slim-bookworm

# Install Git, Git LFS.  How to silence apt-get commands:
# https://peteris.rocks/blog/quiet-and-unattended-installation-with-apt-get/
ARG BRANCH_NAME="master"
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y git git-lfs < /dev/null > /dev/null && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install && \
    git clone --depth 1 --branch "${BRANCH_NAME}" https://github.com/UCL/TLOmodel.git /TLOmodel && \
    rm -rf /root/.ssh

WORKDIR /TLOmodel

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements/dev.txt && \
    pip3 install --no-cache-dir -e .
