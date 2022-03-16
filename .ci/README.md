# Hosting CI

This document describes how to set up continuous integration (CI) for the TLO
project using self-hosted runners (or agents) for GitHub Actions.  This is
currently hosted on an Azure virtual machine running Ubuntu 20.04 LTS.

## Prerequisites

The only prerequisites for hosting the runners are

* [Python](https://www.python.org/).  You need to have in your system the same
  version as that required by the [Tox](https://tox.readthedocs.io/) settings of
  TLO, currently Python 3.6.  Note that Ubuntu 20.04 comes with Python 3.8, so
  you will need to separately install Python 3.6.
* [`systemd`](https://www.freedesktop.org/wiki/Software/systemd/), to set up the
  runners as persistent services.  Other service managers can in principle be
  used, but `systemd` is the default one in recent Ubuntu versions and the
  current CI setup works only with this framework.  Note that to keep the
  service alive after the user running the services logs out you need to enable
  [lingering](https://wiki.archlinux.org/index.php/systemd/User#Automatic_start-up_of_systemd_user_instances)
  for that user
  ```
  sudo loginctl enable-linger <username>
  ```

## Installing the agents

To install the agents you only need to run the *non-interactive* script
[`install_agents.sh`](install_agents.sh) in this directory, with

```sh
./install_agents.sh
```

There are some variables in this script which you may need to manually tweak:

* `NUM_AGENTS`: the number of agents to install
* `SYSTEMD_DIR`: the directory where the `systemd` configuration for the
services will be installed
* `TLO_DIR`: the directory inside which the runners will be installed
* `REPO_URL`: the URL of the TLO repository on GitHub
* `AGENT_VERSION`: the version of the GitHub Actions runners to install, see the
  [release page](https://github.com/actions/runner/releases/latest) to find the
  latest version
* `PYTHON_VER`: the version of Python to be used, currently its value is `3.6`
* `PYTHON`: the absolute path of the Python executable.  Note: this does not
  need to be in the system
  [`PATH`](https://en.wikipedia.org/wiki/PATH_(variable)) environment variable,
  the script will make sure the agents will run with this Python as the first
  Python in their `PATH`
* `PIP_DIR`: the directory where the executable of the
  [`pip`](https://pip.pypa.io/) package manager will be installed.  Also the
  executable of `tox` must be found in this directory.  Note: by default, `pip`
  will be automatically installed if not already present
* `PIP`: the absolute path of the `pip` executable

The script will do the following operations:

* create some symbolic links to the Python executable `PYTHON`, to make sure
  that the agents will find the right version of Python in their `PATH`
* install `pip` for the current user, if not already available
* install `tox` for the current user, if not already available
* install all the GitHub Actions agents, if necessary
* configure the `systemd` service for all agents
* configure the agents, unless they had been already configured before.  This is
  the non-interactive step: for each agent, you will need to manually insert the
  token that you can find in the [Actions settings
  page](https://github.com/UCL/TLOmodel/settings/actions/add-new-runner).  You
  can use the same token for all runners.  You will be prompted for a couple of
  more questions, you can accept the default answer by simply pressing Enter
* (re-)start `systemd` daemon
* (re-)start the `systemd` services running the agents

## `systemd` service configuration

The configuration of the `systemd` services that will run the agents is in the
file [`agent_startup.conf`](agent_startup.conf).  You do not need to modify this
file.

## Other helper scripts

If you want to quickly restart all agents without (re-)installing and
(re-)configuring them, you can run the script

```sh
./restart_agents.sh
```

To stop the services of the running agents you can run the script

```sh
./stop_agents.sh
```
