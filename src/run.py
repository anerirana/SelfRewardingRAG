from rag_pipeline import RAGPipeline, TrainingMode

NUM_TRAIN_EPOCHS = 5
model_config = {
  "NumberOfRetrievedDocuments":8,
  "NumberOfQuerySets":5,
  "NumberOfAugementedQueries":4,
  "NumberOfResponses":3,
  "NumberOfTopkDocuments":10,
  "LanguageModelName":'mistralai/Mistral-7B-Instruct-v0.1', # 'unsloth/mistral-7b-bnb-4bit'
  "CitationModelName":'sentence-transformers/all-mpnet-base-v2',
  "TrainingMode":TrainingMode().SimiliarityScoreCitation
}
rag_pipeline = RAGPipeline(model_config)
original_query = "What is the maximum aggregate principal amount of the commitments provided under this credit agreement?"
doc_ids = ["4c2ec99f83bc81396ff37d5e7abf9880b713a61fc0d6c7b5e1fce184653e226b"]

for epoch in range(NUM_TRAIN_EPOCHS):
    print("=="*20 + " EPOCH " + str(epoch) + "=="*20)
    rag_pipeline.train(original_query,epoch,doc_ids=doc_ids)
