#!/bin/bash
#SBATCH --gpus-per-node=v100:1 # request GPU
#SBATCH --account=def-quiltyjo	# account name
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=84000        # memory per node
#SBATCH --time=02-18:30      # time (DD-HH:MM)
#SBATCH --mail-user=msjahang@uwaterloo.ca	#email id for notification
#SBATCH --mail-type=ALL		#Send me all notifications
#SBATCH --output=NF-Regional-%j.out  # %N for node name, %j for jobID
#SBATCH --job-name="NF-Regional"
# load package
module load python/3.9 cuda cudnn
# source directory of the python file
cd ~/projects/def-quiltyjo/realsina/HDL
# loading virtual environment
source ~/TFP_v1/bin/activate
# running the code
python hpc_regional_NF_v1.py