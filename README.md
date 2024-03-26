# SelfRewardingRAG

## Environement Setup
Steps for setting up a working environment:

1. srun --pty --partition gpu-preempt --constraint a40 --gres=gpu:1 --mem=40GB -t 02:00:00 /bin/bash
2. module load miniconda/22.11.1-1
3. conda create -n SelfRewardRAGEnv python==3.9.7 pip==23.3.1
4. conda activate SelfRewardRAGEnv
5. module load cuda/11.8.0
6. pip install -r requirements.txt
7. pip uninstall --y faiss-cpu & pip install faiss-gpu


NOTE: Steps 1, 2, and, 5 above are specific to unity. To run the code efficiently without unity, a local GPU setup is required.


## Indexing:

`python indexing.py`

NOTE: Uses default indexing location `/scratch/workspace/arana_umass_edu-goldamn_project/credit_agreement_database` accessible to all users in pi group `pi_dhruveshpate_umass_edu` 

## Predictions: 

`python src/run.py`
