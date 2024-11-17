## lab 1
torchrun --nnodes 1 --nproc-per-node 4 ddp.py
mpirun --hostfile hosts -np 4 -x MASTER_ADDR=pink00 -x MASTER_PORT=1234 ./run_training.sh
```
# !/usr/bin/env bash
#
# SBATCH --job-name=bml_lab1
# SBATCH --partition=common
# SBATCH --qos=your_quos
# SBATCH --time=5
# SBATCH --output=output.txt
# SBATCH --nodes=2
# SBATCH --ntasks-per-node=2

export MASTER_PORT=12340
export WORLD_SIZE=${SLURM_NPROCS}

echo "NODELIST="${SLURM_NODELIST}
echo "WORLD_SIZE="${SLURM_NPROCS}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/venv/bin/activate

srun python3 ddp.py
```

## lab 2
to use env variables in terraform variables do something like this
'''
TF_VAR_project=$PROJECT_ID terraform apply
'''

add setup script to bashrc maybe
also these:
alias ssh="ssh -o IdentitiesOnly=yes -F /dev/null -i ~/.ssh/id_gcp"
alias scp="scp -o IdentitiesOnly=yes -F /dev/null -i ~/.ssh/id_gcp"

ssh ~/.ssh/id_gcp kuben_joz@34.56.98.46

scp mpi.py kuben_joz@34.56.98.46:~/mpi.py

export GCP_IP=34.66.117.205
scp mpi.py $GCP_userID@$GCP_IP:~
ssh $GCP_userID@$GCP_IP "mpiexec --hostfile hostfile_mpi -x MASTER_ADDR=bml-0 -x MASTER_PORT=12340 -n 3 python3 mpi.py"

## lab 3
lab3 on cloud
yes yes | ./INIT_CLUSTER.sh

## lab 4
can be local

https://xavierbourretsicotte.github.io/SVM_implementation.html