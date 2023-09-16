

## Overview

A test/staging environment for Github Actions TLOmodel workflows.

The VM has two cores, and we setup two runners for each core to reflect how runners are deployed in live environment.

## Setup

- Fork TLOmodel to your personal account - this is where you'll install the runner.
- Install Vagrant (requires VirtualBox)
- Install Ansible
 
You can use conda environment for Ansible:

```
conda create -n ansible python=3.8
conda activate ansible
conda install -c conda-forge ansible
```

Ansible logs in to the virtual machine using SSH. As you might make/destroy the VM many times, the guest fingerprint changes and then Ansible errors. To prevent this, set:

`export ANSIBLE_HOST_KEY_CHECKING=False`

We need a couple of third-party roles for Ansible. These should be installed before provisioning by running:

```
ansible-galaxy install diodonfrost.git_lfs
ansible-galaxy install monolithprojects.github_actions_runner
```

## Run

To match the self-hosted running VMs deployed on Azure, we use the Ubuntu 20 LTS box:

```
vagrant init ubuntu/jammy64
vagrant up
```

Finally, provision the VM (uses Ansible):

```
vagrant provision
```

(can also be rerun if you change playbook)

`vagrant provision` will use the playbook defined in the Vagrantfile. There is another playbook to install GHA runner, run separately. (There might be a better way to do this.)


The runner requires a [GitHub personal access token](https://github.com/settings/tokens). Use "classic" token and select repo rights. 

Set the GitHub user account and repository where you want to setup the runner (i.e. your personal TLOmodel fork) in provisioning/gha-runner.yml: 

```
github_account: tamuri
github_repo: TLOmodel
```

Then export the token before running Ansible playbook to install the runner:

```
export PERSONAL_ACCESS_TOKEN=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
ansible-playbook -i ./.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory provisioning/gha-runner.yml
```

Once GHA runners have been installed, check they are running:

```
vagrant ssh

# (then in the VM)
systemctl list-units 'actions.runner.*'
```

and check runners on GitHub e.g. for my fork [actions runner setting page](https://github.com/tamuri/TLOmodel/settings/actions/runners)


## Notes

* Some jobs might fail with strange error. I suspect because the vm runs out of memory. Increase in Vagrantfile e.g.

```
    v.memory = 4096
```

then halt/up the vm to refresh.

* Vagrant creates a file for the Ansible inventory. Path is:

.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory

e.g.

`ansible all -m ping -i .vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory`

so, playbooks can be run like so:

```
ansible-playbook provisioning/playbook.yml -i ./.vagrant/provisioners/ansible/inventory/vagrant_ansible_inventory
```

