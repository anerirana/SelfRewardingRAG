from rag_pipeline import RAGPipeline

model_config = {
  "NumberOfRetrievedDocuments":8,
  "NumberOfQuerySets":5,
  "NumberOfAugementedQueries":4,
  "NumberOfResponses":3,
  "NumberOfTopkDocuments":10,
  "BaseModel":"google/flan-t5-xxl"
}
rag_pipeline = RAGPipeline(model_config)
original_query = "Thoroughly examine the given Credit Agreement to identify, summarize, and accentuate the key numerical aspects of any limitations on indebtedness, including conditions allowing borrowers to take on additional debt, any relevant exceptions, and specific numerical figures or values. Clearly cite relevant sections, paragraphs, or clauses, and provide a concise summary highlighting financial metrics and quantitative data."
doc_ids = ["876e2e69-f5cf-4921-bc1f-25304e52db19"]
rag_pipeline.train(original_query, doc_ids)