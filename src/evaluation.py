from rag_pipeline import RAGPipeline, TrainingMode
from os import listdir
import jsonlines
import time
import pandas as pd
from constants import *
import pandas as pd

model_config = { 
  "NumberOfAugementedQueries":5,
  "NumberOfRetrievedDocuments":8,
  "NumberOfTopkDocuments":5,
  "LanguageModelName":'mistralai/Mistral-7B-Instruct-v0.2', # 'Nexusflow/Starling-LM-7B-beta' #'mistralai/Mistral-7B-Instruct-v0.2', # 'unsloth/mistral-7b-bnb-4bit'
  "CitationModelName":'sentence-transformers/all-mpnet-base-v2',
  "TrainingMode":TrainingMode().SimiliarityScoreCitation
}
rag_pipeline = RAGPipeline(model_config)

dir_path = '/scratch/workspace/arana_umass_edu-goldamn_project/quality_analysis_gemini_subset/'
dataset_file_name = 'DatasetQualityAnalysis_Samevg-Data.csv'
df = pd.read_csv(dir_path + dataset_file_name)


start_time = time.time()
doc_ids = [doc_id.split('_response')[0] for doc_id in df['Document_ID']]
gold_answers = df['Response']
original_queries = df['Question']

prompts, pred_answers, rewards, gold_rewards,contri_docs = rag_pipeline.generate_answer(original_queries,doc_ids,gold_answers)
print("="*30 + " Computing Scores " "="*30)
scores = rag_pipeline.compute_scores(gold_answers, pred_answers)
end_time = time.time()
execution_time = (end_time - start_time)/60
print(f"Execution time: {execution_time} minutes")

print("scores:", scores)
result_df = pd.DataFrame({"prompt":prompts,
            "mistral_baseline_answer": pred_answers,
            "mistral_baseline_answer_reward": rewards,
            "mistral_baseline_gold_answer_reward": gold_rewards,
            "mistral_baseline_contri_docs": contri_docs})  
               
pd.concat([df,result_df],axis=1) .to_csv(OUTPUT_DIRECTORY + "mistral_baseline_gemini_subset_results_SS.csv")