# Self-Rewarding RAG

## Installation
```
conda create -n sr-rag python=3.9
conda activate sr-rag

# https://pytorch.org/get-started/locally/
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# This will install src. 
# -e means, if src is edited after running the following command, 
# there's no need to rerun the following pip install again.
pip install -e .
```

## Experiments and Commands
Keep track of the experiments configured under the experiments folder.