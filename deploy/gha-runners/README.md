## Overview

We use GitHub Actions (GHA) self-hosted runners to run TLOmodel tests.
We have many end-to-end tests which take a long time to complete.
Using self-hosted runners means we do not use enterprise account credits and are able to manage the capacity ourselves.

We've written Ansible playbooks, targeting Ubuntu 22.04 LTS (the latest LTS, at time of writing), to install the runners and set up the Python environment.
The playbooks are used to install runners on Azure virtual machines.

You'll also find a Vagrantfile allowing you to use [Vagrant](https://www.vagrantup.com/) to run a self-hosted runner on your own machine, which can be use for testing. The VM has two cores, and we setup two runners for each core to reflect how runners are deployed in live environment.

## Setup

### Using Vagrant for local testing

You can use Vagrant to create a virtual machine on your own machine for quick testing. We have tested on Linux and MacOS. Windows is not supported.

- Fork TLOmodel to your personal account - this is where you'll register the runner.
- Install Vagrant (requires VirtualBox).
- Install Ansible

You can use conda environment for Ansible:

```sh
conda create -n ansible -c conda-forge ansible
conda activate ansible
```

Ansible logs in to the virtual machine using SSH. As you might make/destroy the VM many times, the guest fingerprint changes and then Ansible errors. To prevent this, set:

```sh
export ANSIBLE_HOST_KEY_CHECKING=False
```

We need a collection of third-party roles for Ansible. These should be installed before provisioning by running:

```
ansible-galaxy install -r provisioning/requirements.yml
```

To match the self-hosted running VMs deployed on Azure, we use the Ubuntu 22.04 LTS box:

```sh
vagrant up
```

The Vagrant VM will automatically get provisioned the first time you set it up.
If you need to reprovision the VM after editing the Ansible playbook you can run the command:

```sh
vagrant provision
```

### Using an Azure virtual machine

You do not need Vagrant in this case, only install Ansible as explained in the above section on your own machine.
To create the virtual machines using [Azure CLI](https://learn.microsoft.com/en-gb/cli/azure/) we typically use a command like

```
az vm create --resource-group <RESOURCE_GROUP> --name <NAME> --size Standard_F8s_v2 --location <REGION> --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest --admin-username azureuser --nic-delete-option Delete --data-disk-delete-option Delete --os-disk-delete-option Delete --ssh-key-values <PATH_TO_YOUR_PUBLIC_SSH_KEY>
```

## Run

The runner requires a [GitHub personal access token](https://github.com/settings/tokens). Use "classic" token and select repo rights.

Set the GitHub user account and repository where you want to setup the runner in `provisioning/playbook.yml`.
By default we have

```yaml
github_account: UCL
github_repo: TLOmodel
```

but you can point it to your personal TLOmodel fork for testing.

Then export the token before running Ansible playbook to install the runner:

```sh
export PERSONAL_ACCESS_TOKEN=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
ansible-playbook -i <hostname-or-ip-address>, -u azureuser provisioning/playbook.yml
```

The argument to `-i` can be either a comma-separated list of hosts where to run the playbook on (this list has to end with a command if you want to run the playbook on a single host, hosts can be specified by IP addresses or hosts names defined in your local SSH or network configurations) or the path to the Ansible inventory you need to access the virtual machine, either a local one or the remote Azure one.
The argument to `-u` is the user of the machine where to run the playbook on, it can be empty if it is the same as the current user.

There are some variables in the playbook that are intended to be [set at runtime](https://docs.ansible.com/ansible/latest/playbook_guide/playbooks_variables.html#defining-variables-at-runtime):

* `n_runners`: the number of runners you want to install (default if not specified: `1`)
* `extra_runner_labels`: the labels to add to the GHA runners, as a list of strings (default if not specified: `[]`)

These variables can be set with the `--extra-vars` argument.
Note that lists (such as `extra_runner_labels`) have to be set with the JSON format.
Here are some examples:

```sh
ansible-playbook -i <hostname-or-ip-address>, -u azureuser provisioning/playbook.yml --extra-vars "n_runners=2" # Install 2 runners
ansible-playbook -i <hostname-or-ip-address>, -u azureuser provisioning/playbook.yml --extra-vars '{"extra_runners_labels": ["sim"]}' # Install 1 runner with label "sim"
ansible-playbook -i <hostname-or-ip-address>, -u azureuser provisioning/playbook.yml --extra-vars '{"n_runners": "8", "extra_runners_labels": ["test"]}' # Install 8 runners with label "test"
```

Once GHA runners have been installed, check they are running:

```sh
vagrant ssh         # only if you're using Vagrant for local testing
ssh XXX.XXX.XXX.XXX # if you're logging into the Azure virtual machine

# (then in the VM)
systemctl list-units 'actions.runner.*'
```

and check runners on GitHub in the [actions runner setting page](https://github.com/UCL/TLOmodel/settings/actions/runners).

### Simulation runner machine

For the machine running the simulations, run the playbook `provisioning/task-runner-playbook.yml`, which automatically runs the playbook `provisioning/playbook.yml` and some extra operations needed only on these machines (e.g. mounting external storage volumes).
The playbook `task-runner-playbook.yml` does not require extra arguments, and it automatically forwards all arguments to `playbook.yaml`.
Example:

```sh
ansible-playbook -i <hostname-or-ip-address>, -u azureuser provisioning/task-runner-playbook.yml --extra-vars '{"extra_runners_labels": ["sim", "tasks"], "n_runners":"15"}'
```

> [!NOTE]
> On the simulation runner machine we usually want to install $n$ GitHub Actions runners, $n-1$ of which should have the `tasks` label, and the other one the `postprocess` label.
> We do not have an easy way to create different sets of runners with different labels, for the time being the simplest option is to set up all runners with the `tasks` label, and manually change the label of one to `postprocess` in GitHub web UI.

## Notes

### Vagrant

* Some jobs might fail with strange error. I suspect because the vm runs out of memory. Increase in Vagrantfile e.g.

```
    v.memory = 4096
```

then halt/up the vm to refresh.

* Vagrant creates a file for the Ansible inventory. Path is:

```
.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory
```

e.g.

```sh
ansible all -m ping -i .vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory
```

so, playbooks can be run like so:

```sh
ansible-playbook provisioning/playbook.yml -i ./.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory
```
