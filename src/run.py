from rag_pipeline import RAGPipeline, TrainingMode
from os import listdir
import jsonlines
import time

NUM_TRAIN_EPOCHS = 3
model_config = {
  "NumberOfRetrievedDocuments":5,
  "NumberOfQuerySets":5,
  "NumberOfAugementedQueries":5,
  "NumberOfResponses":5,
  "NumberOfTopkDocuments":5,
  "LanguageModelName":'mistralai/Mistral-7B-Instruct-v0.2', # 'unsloth/mistral-7b-bnb-4bit'
  "CitationModelName":'sentence-transformers/all-mpnet-base-v2',
  "TrainingMode":TrainingMode().SimiliarityScoreCitation
}
rag_pipeline = RAGPipeline(model_config)

queries_dir = "/scratch/workspace/arana_umass_edu-goldamn_project/data/jsonls_eval/"
doc_ids = [f.split('.')[0] for f in listdir(queries_dir)]
original_queries = []
for doc_id in doc_ids:
    row = []
    with jsonlines.open(queries_dir + doc_id + ".jsonl") as f:
        for line in f.iter():
            row.append(line['question'])
    original_queries.append(row)

# original_query = "What is the maximum aggregate principal amount of the commitments provided under this credit agreement?"
# doc_ids = ["4c2ec99f83bc81396ff37d5e7abf9880b713a61fc0d6c7b5e1fce184653e226b"]

for epoch in range(NUM_TRAIN_EPOCHS):
    start_time = time.time()
    print("=="*20 + " EPOCH " + str(epoch) + "=="*20)
    rag_pipeline.train(original_queries,epoch,doc_ids=doc_ids)
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} minutes")