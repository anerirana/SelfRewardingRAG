from rag_pipeline import RAGPipeline

model_config = {
  "NumberOfRetrievedDocuments":8,
  "NumberOfQuerySets":5,
  "NumberOfAugementedQueries":4,
  "NumberOfResponses":3,
  "NumberOfTopkDocuments":10,
  "BaseModel":"mistralai/Mistral-7B-Instruct-v0.2"
}
rag_pipeline = RAGPipeline(model_config)
original_query = "Thoroughly examine the given Credit Agreement to identify, summarize, and accentuate the key numerical aspects of any limitations on indebtedness, including conditions allowing borrowers to take on additional debt, any relevant exceptions, and specific numerical figures or values. Clearly cite relevant sections, paragraphs, or clauses, and provide a concise summary highlighting financial metrics and quantitative data."
oq2 = "What is the maximum aggregate principal amount of the commitments provided under this credit agreement?"
rag_pipeline.train(oq2)
