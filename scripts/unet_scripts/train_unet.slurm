#SBATCH -J unet_lane                # Job name
#SBATCH --time=3-00:00:00             # 3 day wall-clock
#SBATCH --nodes=1                  # one node is enough for 1 GPU
#SBATCH --ntasks=10                 # 10 CPU cores total  (8 dataloader workers)
#SBATCH --gres=gpu:l40s:1          # 1 L40S GPU  (change to a30 etc.)
#SBATCH --mem=90G                    # all mem on the node is mem=0 but long queue (or set e.g. 120G)
#SBATCH -o /scratch/dannycharm-alt-REU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu
