## Overview

A test/staging environment for Github Actions TLOmodel workflows.

The VM has two cores, and we setup two runners for each core to reflect how runners are deployed in live environment.

## Setup

We will deploy the runners on Azure virtual machines, but you can also use Vagrant to create virtual machines on your own computer for local testing.

### Using Vagrant for local testing

- Fork TLOmodel to your personal account - this is where you'll install the runner.
- Install Vagrant (requires VirtualBox)
- Install Ansible
 
You can use conda environment for Ansible:

```sh
conda create -n ansible python=3.8
conda activate ansible
conda install -c conda-forge ansible
```

Ansible logs in to the virtual machine using SSH. As you might make/destroy the VM many times, the guest fingerprint changes and then Ansible errors. To prevent this, set:

```sh
export ANSIBLE_HOST_KEY_CHECKING=False
```

To match the self-hosted running VMs deployed on Azure, we use the Ubuntu 22.04 LTS box:

```sh
vagrant up
```

Finally, provision the VM (uses Ansible):

```sh
vagrant provision
```

(can also be rerun if you change playbook)

`vagrant provision` will use the playbook defined in the Vagrantfile. 

### Using an Azure virtual machine

You do not need Vagrant, only install Ansible as explained in the above section on your own machine.

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
ansible-playbook -i ./.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory provisioning/gha-runner.yml --extra-vars "n_runners=2"
```

where `n_runners` has to be set to the number of runners you want to install.
The argument to `-i` is the path to the Ansible inventory you need to access the virtual machine, either a local one or the remote Azure one.

Once GHA runners have been installed, check they are running:

```sh
vagrant ssh         # only if you're using Vagrant for local testing
ssh XXX.XXX.XXX.XXX # if you're logging into the Azure virtual machine

# (then in the VM)
systemctl list-units 'actions.runner.*'
```

and check runners on GitHub in the [actions runner setting page](https://github.com/UCL/TLOmodel/settings/actions/runners).


## Notes

### Vagrant

* Some jobs might fail with strange error. I suspect because the vm runs out of memory. Increase in Vagrantfile e.g.

```
    v.memory = 4096
```

then halt/up the vm to refresh.

* Vagrant creates a file for the Ansible inventory. Path is:

.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory

e.g.

```sh
ansible all -m ping -i .vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory
```

so, playbooks can be run like so:

```sh
ansible-playbook provisioning/playbook.yml -i ./.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory
```
