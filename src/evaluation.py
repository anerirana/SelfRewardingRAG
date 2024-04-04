from rag_pipeline import RAGPipeline, TrainingMode
from os import listdir
import jsonlines
import time
import pandas as pd
from constants import *

model_config = {
  "NumberOfTopkDocuments":5,
  "LanguageModelName":'mistralai/Mistral-7B-Instruct-v0.2', # 'Nexusflow/Starling-LM-7B-beta' #'mistralai/Mistral-7B-Instruct-v0.2', # 'unsloth/mistral-7b-bnb-4bit'
  "CitationModelName":'sentence-transformers/all-mpnet-base-v2',
  "TrainingMode":TrainingMode().SimiliarityScoreCitation
}
rag_pipeline = RAGPipeline(model_config)

queries_dir = "/scratch/workspace/arana_umass_edu-goldamn_project/data/jsonls_eval/"
doc_ids = [f.split('.')[0] for f in listdir(queries_dir)]
original_queries = []
gold_answers = []
for doc_id in doc_ids:
    q_row = []
    with jsonlines.open(queries_dir + doc_id + ".jsonl") as f:
        for line in f.iter():
            q_row.append(line['question'])
            gold_answers.append([line['answer']])
        original_queries.append(q_row)

start_time = time.time()
prompts, pred_answers, rewards, gold_rewards,contri_docs = rag_pipeline.generate_answer(original_queries,doc_ids,gold_answers)
print("="*30 + " Computing Scores " "="*30)
scores = rag_pipeline.compute_scores(gold_answers, pred_answers)
end_time = time.time()
execution_time = (end_time - start_time)/60
print(f"Execution time: {execution_time} minutes")

print("scores:", scores)
df = pd.DataFrame({"prompts":prompts,
            "gold_answer":gold_answers,
            "pred_answer":pred_answers,
            "reward":rewards,
            "gold_rewards":gold_rewards,
            "contri_docs": contri_docs})           
df.to_csv(OUTPUT_DIRECTORY + "mistral_baseline_aug_results.csv")