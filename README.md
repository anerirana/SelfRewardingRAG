# SelfRewardingRAG

## Environement Setup
Steps for setting up a working environment:

1. srun --pty --partition gpu-preempt --constraint a40 --gres=gpu:1 --mem=40GB -t 02:00:00 /bin/bash
2. module load miniconda/22.11.1-1
3. conda create -n SelfRewardRAGEnv python==3.9.7 pip==23.3.1
4. conda activate SelfRewardRAGEnv
5. module load cuda/11.8.0
6. pip install -r requirements.txt
7. pip install faiss-gpu & pip uninstall --y faiss-cpu

<!--- When using unsloth for PEFT DPO training--->
<!--- pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"--->

NOTE: Steps 1, 2, and, 5 above are specific to unity. To run the code efficiently without unity, a local GPU setup is required.


## Indexing:

```
usage: python indexing.py [-h] [--new_index] [--not-new_index] dir_path index_path

Process input directory

positional arguments:
  dir_path         The path to the directory containing credit agreements
  index_path       The path to the index for the credit agreement database

optional arguments:
  --new_index      Flag to indicate if a new index should be created (default: True)
  --not-new_index  Flag to indicate if a new index should not be created
```

To increase the speed of indexing, a few small changes were made to the ragatouille/data/preprocessors.py file. Multiprocessing support was added to the llama_index_sentence_splitter function.
The following code was added instead of the commented code.

```
    import multiprocess as mp

    def process_docs(doc_id, doc):
        chunks = [
          {"document_id": doc_id, "content": node.text} for node in node_parser(doc)
        ]
        return chunks
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_docs, zip(document_ids, docs))
        chunks = [chunk for sublist in results for chunk in sublist]
    # for doc_id, doc in zip(document_ids, docs):
    #     chunks += [
    #         {"document_id": doc_id, "content": node.text} for node in node_parser(doc)
    #     ]
```


## Training: 

1. Confirgure training parameters in `training.py`. Following are the default values:-

Parameter | Default Value | Description 
--- | --- | --- 
NumberOfRetrievedDocuments | 5 | The 'p' number of docouments retrieved for each augmented query.
NumberOfQuerySets | 5 | The 'p' number of documents retrieved for each augmented query.
NumberOfQuerySets | 5 | The 'm' number of query augmentation sets created for a given original query.
NumberOfAugementedQueries | 5 | The 'n' number of augmented queries created in each set for a given original query.
NumberOfResponses | 5 | The 'l' number of answers to generate for a given RAG prompt
NumberOfTopkDocuments | 5 | The 'k' number of top documents selected to create a RAG prompt
LanguageModelName | 'mistralai/Mistral-7B-Instruct-v0.2' | The pre-trained language model for fine-tuning
CitationModelName | 'sentence-transformers/all-mpnet-base-v2' | A senetence trasnfromer model to calcualte cosine similarity scores. Only applicable when TrainingMode is 'SimiliarityScoreCitation'  
TrainingMode | TrainingMode().ResponseWithCitation | To perform experiments with different training modes i.e. 'ResponseWithCitation' and 'SimiliarityScoreCitation'
QueryAugmentationBatchSize | 16 | Batch Size for query augmentation generation.
AnswerGenerationBtachSize | 8 |  Batch Size for retrieval augemented generation.
RewardGenerationBtachSize | 8 | Batch Size for reward prediction on generated answers.

2. Edit the `constant.py` file. Following  parameters are REQUIRED to be customized before training.

NOTE: You can also edit prompts for each task in the RAG pipeline here

Parameter | Description
--- | --- 
TRANSFORMERS_TOKEN | Create a user access token [here](https://huggingface.co/docs/hub/en/security-tokens).
OUTPUT_DIRECTORY | Provide the path to an output directory for saving output and error files from training and evaluation.
TRAIN_OUTPUT_DIRECTORY | Provide the path to a directory for saving the fine tuned model, its checkpoints and configuration.
PATH_TO_INDEX | Provide the path to the Ragatouille index directory.

4. Run the training code
`python src/training.py`

## Evaluation:

`python src/evaluation.py`

