#!/bin/bash

#echo 'alias ssh="ssh -o IdentitiesOnly=yes -F /dev/null -i ~/.ssh/id_gcp"\n' >> /etc/bash.bashrc
#echo 'alias scp="scp -o IdentitiesOnly=yes -F /dev/null -i ~/.ssh/id_gcp"\n' >> /etc/bash.bashrc
printf '%s\n' '. /workspaces/BML-LML/setup.sh' >> ~/.bashrc

cp /workspaces/BML-LML/.ssh/config ~/.ssh/config

#gcloud
sudo apt-get -y update
sudo apt-get -y install apt-transport-https ca-certificates gnupg curl vim
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get -y update && sudo apt-get -y install google-cloud-cli

#terraform
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt -y update && sudo apt -y install terraform

#ansible
pip install --user -r requirements.txt