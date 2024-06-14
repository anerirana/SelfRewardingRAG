from rag_pipeline import RAGPipeline, TrainingMode
from os import listdir
import jsonlines
import time
import pandas as pd
import numpy as np
from constants import *

model_config = { 
  "NumberOfRetrievedDocuments":5, #p
  "NumberOfQuerySets":1, #m
  "NumberOfAugementedQueries":4,#n
  "NumberOfResponses":1, #l
  "NumberOfTopkDocuments":4, #k
  "LanguageModelName": TRAIN_OUTPUT_DIRECTORY + '/mistral-v2-finetuned_0', #'mistralai/Mistral-7B-Instruct-v0.2', # 'Nexusflow/Starling-LM-7B-beta' #'mistralai/Mistral-7B-Instruct-v0.2', # 'unsloth/mistral-7b-bnb-4bit'
  "CitationModelName":'sentence-transformers/all-mpnet-base-v2',
  "TrainingMode":TrainingMode().ResponseWithCitation,
  "QueryAugmentationBatchSize": 16,
  "AnswerGenerationBtachSize": 8,
  "RewardGenerationBtachSize": 8
}
rag_pipeline = RAGPipeline(model_config)

# Uncomment this section to read all the 400 docsument quesries
# queries_dir = "/scratch/workspace/arana_umass_edu-goldamn_project/data/jsonls_eval/"
# doc_ids = [f.split('.')[0] for f in listdir(queries_dir)]
# original_queries = []
# gold_answers = []
# for doc_id in doc_ids:
#     q_row = []
#     with jsonlines.open(queries_dir + doc_id + ".jsonl") as f:
#         for line in f.iter():
#             q_row.append(line['question'])
#             gold_answers.append([line['answer']])
#         original_queries.append(q_row)
# prompts, pred_answers, rewards, gold_rewards,contri_docs = rag_pipeline.generate_answer(original_queries,doc_ids,gold_answers)

start_time = time.time()

df = pd.read_csv("~/SelfRewardingRAG/data/mistral_basdeline_human_reward_annotations.csv")[:100]
print("df.shape", df.shape)

doc_ids = np.asarray(df['Document_ID'].apply(lambda x: x.split('_response')[0]))
gold_answers = np.asarray(df['Response'])

original_queries = np.asarray(df['Question'])
print("len(original_queries): ", len(original_queries))
print("len(doc_ids): ", len(doc_ids))
print("len(gold_answers): ", len(gold_answers))

prompts, pred_answers, _, _, contri_docs = rag_pipeline.prediction(original_queries,doc_ids,gold_answers)
print("len(prompts): ", len(prompts))
print("len(pred_answers): ", len(pred_answers))
print("len(contri_docs): ", len(contri_docs))
# print("len(rewards): ", len(rewards))
# print("len(gold_rewards): ", len(gold_rewards))
# print("="*30 + " Computing Scores " + "="*  30)
# scores = rag_pipeline.compute_scores(gold_answers, pred_answers)
end_time = time.time()
execution_time = (end_time - start_time)/60
print(f"Execution time: {execution_time} minutes")

# print("scores:", scores)
df = pd.DataFrame({"doc_ids":doc_ids,
                   "original_query": original_queries,
            "prompt":prompts,
            "gold_answer":gold_answers,
            "baseline_answer":df['mistral_baseline_answer'],
            "fine_tuned_answer": pred_answers,
            # "baseline_answer_model_reward":rewards,
            # "gold_answer_model_reward":gold_rewards,
            # "contri_docs": contri_docs
            })           
df.to_csv(OUTPUT_DIRECTORY + "mistral_ft_train_results.csv")