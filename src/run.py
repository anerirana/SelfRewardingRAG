from rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline({})
original_query = "Thoroughly examine the given Credit Agreement to identify, summarize, and accentuate the key numerical aspects of any limitations on indebtedness, including conditions allowing borrowers to take on additional debt, any relevant exceptions, and specific numerical figures or values. Clearly cite relevant sections, paragraphs, or clauses, and provide a concise summary highlighting financial metrics and quantitative data."
rag_pipeline.train(original_query)