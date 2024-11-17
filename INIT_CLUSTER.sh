#!/bin/bash
echo '' > ~/.ssh/known_hosts
yes yes | terraform apply
python parse-tf-state.py
ansible-playbook -i hosts install_packages.yml
ansible-playbook -i hosts config_ssh.yml
ansible-playbook -i hosts nfs.yml
