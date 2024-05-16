FROM python:3.11-slim-bookworm

# Setup SSH
RUN mkdir /root/.ssh
COPY id_ed25519 /root/.ssh/id_ed25519
COPY known_hosts /root/.ssh/known_hosts
RUN chmod -R 0600 /root/.ssh

# Install Git, Git LFS.  How to silence apt-get commands:
# https://peteris.rocks/blog/quiet-and-unattended-installation-with-apt-get/
RUN apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -qq -y git git-lfs < /dev/null > /dev/null
# Cleanup apt cache
RUN rm -rf /var/lib/apt/lists/*
# Configure git lfs
RUN git lfs install

# Clone the TLOModel repository and move in it
ARG BRANCH_NAME="master"
RUN git clone --branch "${BRANCH_NAME}" git@github.com:UCL/TLOmodel.git /TLOmodel
WORKDIR /TLOmodel
RUN git gc --aggressive

# Install dependencies
RUN pip3 install -r requirements/dev.txt
RUN pip3 install -e .
