import random

QUERY_AUGMENTATION_PROMPT = "Give {n} different versions of the query: {original_query}"

RAG_PROMPT = "Answer the following question: {original_query} using information from the following documents: {documents}. Highlight which documents you used to answer the question."

DECODE_PARAMS_DICT = {
            "temperature":random.choice([0.1,0.2,0.3,0.4]), 
            "top_p":random.choice([0.99, 0.8, 0.7, 0.6, 0.5]), 
            "repetition_penalty":random.choice([1.2, 1.3, 1.4, 1.5]), 
            "min_new_tokens":random.choice([16, 32, 64, 128]), 
            "max_new_tokens":random.choice([2016, 2032, 2064, 2128])
          }