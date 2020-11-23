#!/bin/bash

set -e

NUM_AGENTS=4

SRC_DIR=$(realpath "$( dirname "${BASH_SOURCE[0]}" )")
SYSTEMD_DIR="${HOME}/.config/systemd/user"
TLO_DIR="${HOME}/tlo"
BIN_DIR="${TLO_DIR}/bin"

AGENTS_DIR="${TLO_DIR}/gha-agents"
REPO_URL="https://github.com/UCL/TLOmodel"
AGENT_VERSION="2.274.2"
AGENT_URL="https://github.com/actions/runner/releases/download/v${AGENT_VERSION}/actions-runner-linux-x64-${AGENT_VERSION}.tar.gz"

PYTHON_VER=3.6
PYTHON="/usr/bin/python${PYTHON_VER}"
PIP_DIR="${HOME}/.local/bin"
PIP="${PIP_DIR}/pip${PYTHON_VER}"
AGENT_PATH="${BIN_DIR}:${PIP_DIR}:${PATH}"

# Prepare systemd directory
mkdir -p "${SYSTEMD_DIR}"
rm -f "${SYSTEMD_DIR}"/tlo_gha_agent_*.service

# Create symlinks for Python
echo -n "Creating Python ${PYTHON_VER} symlinks..."
mkdir -p "${BIN_DIR}"
rm -f "${BIN_DIR}"/*
ln -s "${PYTHON}" "${BIN_DIR}/python${PYTHON_VER}"
ln -s "python${PYTHON_VER}" "${BIN_DIR}/python3"
ln -s "python3" "${BIN_DIR}/python"
echo "done"

# Install pip
if [[ ! -f "${PIP}" ]]; then
    TMPDIR=$(mktemp -d)
    curl https://bootstrap.pypa.io/get-pip.py -o "${TMPDIR}/get-pip.py"
    "${PYTHON}" "${TMPDIR}/get-pip.py" --user
    rm -rf "${TMPDIR}"
fi
# Install tox
if [[ ! -f "${PIP_DIR}/tox" ]]; then
    "${PIP}" install --user tox
fi

for AGENT_IDX in $(seq 1 ${NUM_AGENTS}); do
    AGENT_DIR="${AGENTS_DIR}/${AGENT_IDX}"

    # Install agent executable
    if [[ ! -d "${AGENT_DIR}" ]]; then
        echo -n "Installing agent #${AGENT_IDX}..."
        mkdir -p "${AGENT_DIR}"
        curl -LsS "${AGENT_URL}" | tar -xz -C "${AGENT_DIR}"
        echo "done"
    fi
    # Copy the script bin/runsvc.sh to the root directory of the agent, because
    # it internally uses paths relative to that directory (`svc.sh` does the
    # same move).
    cp "${AGENT_DIR}/bin/runsvc.sh" "${AGENT_DIR}/runsvc.sh"
    # Set the path for the agents
    echo "${AGENT_PATH}" > "${AGENT_DIR}/.path"

    # Create systemd configuration
    echo -n "Configuring systemd service for agent #${AGENT_IDX}..."
    export AGENT_IDX AGENT_PATH AGENT_DIR
    envsubst '$AGENT_IDX $AGENT_PATH $AGENT_DIR'  <"agent_startup.conf" >"${SYSTEMD_DIR}/tlo_gha_agent_${AGENT_IDX}.service"
    echo "done"

    # Configure the agent
    if [[ ! -f "${AGENT_DIR}/.credentials" ]]; then
        /bin/bash -c "cd ${AGENT_DIR}; ./config.sh --name ${HOSTNAME}.${AGENT_IDX} --url ${REPO_URL} --replace"
    fi
done

# Reload systemd user daemon
systemctl --user daemon-reload

echo -n "Restarting agents..."
for AGENT_IDX in $(seq 1 ${NUM_AGENTS}); do
    systemctl --user stop tlo_gha_agent_${AGENT_IDX} || true
    # Enable and start GHA agents
    systemctl --user enable tlo_gha_agent_${AGENT_IDX}
    systemctl --user start tlo_gha_agent_${AGENT_IDX}
done
echo "done"
